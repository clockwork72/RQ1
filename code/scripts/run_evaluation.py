#!/usr/bin/env python3
"""End-to-end evaluation runner.

Re-runs the three appendix evaluations against an OpenAI-compatible LLM
endpoint (`LLM_BASE_URL`) and writes per-run JSONs whose shape matches the
files bundled in `data/dataset.tar.gz` under `benchmarks/`.

Subcommands
-----------
perturbation
    Re-runs the verifier perturbation eval (Evaluation §4.2.1). Reads the
    100 synthetic clause pairs from
    `data/raw/benchmarks/perturbation_cases.jsonl`, asks the verifier
    (current `VERIFIER_PROMPT` from `code/prompts/unified_prompts.py`) for
    a verdict on each, and computes 3-class precision / recall / F1 plus
    the binary view used in the paper.

leaderboard
    Re-runs one row of the extractor leaderboard (Evaluation §4.1) on the
    100-clause holdout in `data/raw/benchmarks/holdout_clauses_100.jsonl`,
    matches the predicted PPSes against
    `data/raw/benchmarks/gold_claude_holdout_100_v3.jsonl` with the same
    score-greedy bipartite matcher used in the paper (action-compatibility
    + data-object Jaccard + recipient Jaccard, threshold 0.40), and writes
    strict + adjusted P / R / F1.

agreement
    Re-runs one model's column of the verdict-agreement eval (Evaluation
    §4.2.2). Reads the 100 inconsistent findings sampled with seed 42
    from `data/raw/benchmarks/verdict_agreement_findings.jsonl`, asks the
    verifier for a verdict on each, and writes a per-model verdicts JSON.
    Re-running this with a different served model gives you the
    corresponding column of the agreement table; use several columns
    together to reproduce the pairwise-agreement numbers.

Common flags
------------
--base-url / env LLM_BASE_URL   default http://localhost:8000/v1
--api-key  / env LLM_API_KEY    default "local"
--model    / env LLM_MODEL      default gemma3:27b
--out                           output path (per-task default below)
--max-cases                     cap for smoke tests; default = full set

Quick start
-----------

    bash code/llm_serving/serve_vllm.sh                 # local 2x A100, or rented
    source code/llm_serving/env_local.sh                # exports LLM_BASE_URL etc.

    python code/scripts/run_evaluation.py perturbation --out out/eval_perturbation.json
    python code/scripts/run_evaluation.py leaderboard  --out out/eval_leaderboard.json
    python code/scripts/run_evaluation.py agreement    --out out/eval_agreement.json

Each subcommand prints a one-line summary at the end so you can compare
against the paper's numbers without opening the JSON.

Notes on reproducibility
------------------------
* Same numbers as the paper require the same model checkpoint. Pin the
  vLLM / Ollama version and the model weights you serve.
* LLMs are not byte-deterministic across server versions even at
  temperature=0; expect <=1% drift from the paper's binary metrics.
* The verifier's emit vocabulary is `inconsistent | unspecified |
  non_conflict`. The bundled ground-truth labels use the paper-text
  vocabulary `inconsistent | underspecified | non_conflict`. The metrics
  code below treats `unspecified` and `underspecified` as the same class,
  so the binary view is identical and the 3-class view is unaffected.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai (>=1.0)", file=sys.stderr)
    sys.exit(2)

from prompts.unified_prompts import EXTRACTION_PROMPT, VERIFIER_PROMPT

DATA_DIR = REPO_ROOT / "data" / "raw" / "benchmarks"
VERDICTS = ["inconsistent", "underspecified", "non_conflict"]


def _norm_verdict(v: str) -> str:
    v = (v or "").strip().lower()
    if v == "unspecified":
        return "underspecified"
    if v == "soft_tension":
        return "underspecified"
    if v == "hard_contradiction":
        return "inconsistent"
    return v


def get_client(args) -> tuple[OpenAI, str]:
    base_url = args.base_url or os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
    api_key  = args.api_key  or os.environ.get("LLM_API_KEY", "local")
    model    = args.model    or os.environ.get("LLM_MODEL", "gemma3:27b")
    print(f"[run_evaluation] endpoint = {base_url}")
    print(f"[run_evaluation] model    = {model}")
    return OpenAI(base_url=base_url, api_key=api_key, timeout=120), model


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def call_chat(client: OpenAI, model: str, prompt: str, max_retries: int = 3,
              temperature: float = 0.0, max_tokens: int = 2048) -> str:
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            print(f"  retry {attempt + 1}/{max_retries} after error: {exc}")
            time.sleep(backoff)
            backoff *= 2
    return ""


def parse_verdict_response(text: str) -> dict | None:
    """Production parser: try direct JSON, then code-fence, then first {...}."""
    if not text:
        return None
    content = text.strip()
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "verdict" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, dict) and "verdict" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict) and "verdict" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
    return None


def parse_json_array(text: str) -> list[dict]:
    if not text:
        return []
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    except json.JSONDecodeError:
        pass
    m = re.search(r'\[[\s\S]*\]', text)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
        except json.JSONDecodeError:
            pass
    return []


# ---------------------------------------------------------------------------
# Verifier prompt rendering (matches the paper's run)
# ---------------------------------------------------------------------------

PATTERN_NAMES = {
    "Π₁": "Modality Contradiction",
    "Π₂": "Exclusivity Violation",
    "Π₃": "Condition Asymmetry",
    "Π₄": "Temporal Contradiction",
    "Π₅": "Temporal Contradiction",
    "Π₇": "Vagueness Asymmetry",
    "Π₈": "Commitment Transitivity Violation",
    "Π₉": "Rights Propagation Failure",
}


def _field(stmt: dict, key: str, default: str = "unspecified") -> str:
    val = str(stmt.get(key, "")).strip()
    return val if val and val.lower() not in {"", "none", "null", "unspecified"} else default


def _policy_label(stmt: dict) -> str:
    ps = str(stmt.get("policy_source", "first_party"))
    if "third_party" in ps or "vendor" in ps:
        vendor = ps.split(":", 1)[-1] if ":" in ps else "vendor"
        return f"vendor policy ({vendor})"
    return "website policy"


def build_verifier_prompt(s1: dict, s2: dict, *, pattern_id: str = "",
                          pattern_explanation: str = "",
                          context_1: str = "", context_2: str = "") -> str:
    src1 = str(s1.get("source_text", ""))[:800]
    src2 = str(s2.get("source_text", ""))[:800]
    return VERIFIER_PROMPT.format(
        pattern_id=pattern_id,
        pattern_name=PATTERN_NAMES.get(pattern_id, pattern_id),
        pattern_description="",
        policy_1_label=_policy_label(s1),
        source_text_1=src1,
        context_1=context_1 or src1,
        actor_1=_field(s1, "actor"),
        action_1=_field(s1, "action"),
        modality_1=_field(s1, "modality"),
        data_1=_field(s1, "data_object"),
        purpose_1=_field(s1, "purpose"),
        condition_1=_field(s1, "condition"),
        policy_2_label=_policy_label(s2),
        source_text_2=src2,
        context_2=context_2 or src2,
        actor_2=_field(s2, "actor"),
        action_2=_field(s2, "action"),
        modality_2=_field(s2, "modality"),
        data_2=_field(s2, "data_object"),
        purpose_2=_field(s2, "purpose"),
        condition_2=_field(s2, "condition"),
        pattern_explanation=str(pattern_explanation)[:400],
        neighborhood_context="",
    )


# ---------------------------------------------------------------------------
# Perturbation
# ---------------------------------------------------------------------------

def compute_perturbation_metrics(predictions: list[dict]) -> dict:
    correct = sum(1 for p in predictions if p["predicted"] == p["ground_truth"])
    n = len(predictions)
    accuracy = correct / n if n else 0.0
    confusion = {gt: {pred: 0 for pred in VERDICTS} for gt in VERDICTS}
    tp = Counter(); fp = Counter(); fn = Counter()
    for p in predictions:
        g, q = p["ground_truth"], p["predicted"]
        if g == q:
            tp[g] += 1
        else:
            fp[q] += 1
            fn[g] += 1
        if g in confusion and q in confusion[g]:
            confusion[g][q] += 1
    per_class = {}
    for v in VERDICTS:
        prec = tp[v] / (tp[v] + fp[v]) if (tp[v] + fp[v]) else 0.0
        rec  = tp[v] / (tp[v] + fn[v]) if (tp[v] + fn[v]) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[v] = {"precision": round(prec, 4), "recall": round(rec, 4),
                        "f1": round(f1, 4), "support": tp[v] + fn[v]}
    macro_f1 = sum(c["f1"] for c in per_class.values()) / len(per_class)

    def _bin(v): return "non_conflict" if v == "non_conflict" else "genuine"
    btp = sum(1 for p in predictions if _bin(p["predicted"]) == "genuine"      and _bin(p["ground_truth"]) == "genuine")
    bfp = sum(1 for p in predictions if _bin(p["predicted"]) == "genuine"      and _bin(p["ground_truth"]) == "non_conflict")
    bfn = sum(1 for p in predictions if _bin(p["predicted"]) == "non_conflict" and _bin(p["ground_truth"]) == "genuine")
    btn = sum(1 for p in predictions if _bin(p["predicted"]) == "non_conflict" and _bin(p["ground_truth"]) == "non_conflict")
    bp = btp / (btp + bfp) if (btp + bfp) else 0.0
    br = btp / (btp + bfn) if (btp + bfn) else 0.0
    bf = 2 * bp * br / (bp + br) if (bp + br) else 0.0
    bacc = (btp + btn) / n if n else 0.0
    bp_nc = btn / (btn + bfn) if (btn + bfn) else 0.0
    br_nc = btn / (btn + bfp) if (btn + bfp) else 0.0
    bf_nc = 2 * bp_nc * br_nc / (bp_nc + br_nc) if (bp_nc + br_nc) else 0.0
    macro_f1_binary = (bf + bf_nc) / 2

    return {
        "accuracy_3class": round(accuracy, 4),
        "macro_f1_3class": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_3class": confusion,
        "accuracy_binary": round(bacc, 4),
        "macro_f1_binary": round(macro_f1_binary, 4),
        "binary_confusion": {"genuine": {"genuine": btp, "nc": bfn},
                             "nc":      {"genuine": bfp, "nc": btn}},
    }


def cmd_perturbation(args):
    client, model = get_client(args)
    cases_path = Path(args.cases) if args.cases else DATA_DIR / "perturbation_cases.jsonl"
    if not cases_path.exists():
        print(f"ERROR: missing {cases_path}. Extract data/dataset.tar.gz first.", file=sys.stderr)
        sys.exit(2)
    cases = [json.loads(l) for l in cases_path.open()]
    if args.max_cases:
        cases = cases[: args.max_cases]
    cases = [{**c, "ground_truth": _norm_verdict(c["ground_truth"])} for c in cases]
    gt_dist = dict(Counter(c["ground_truth"] for c in cases))
    print(f"[perturbation] {len(cases)} cases (gt: {gt_dist})")

    predictions = []
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        prompt = build_verifier_prompt(
            case["statement_1"], case["statement_2"],
            pattern_id=case.get("pattern_id", ""),
            pattern_explanation=case.get("explanation", ""),
        )
        try:
            text = call_chat(client, model, prompt)
        except Exception as exc:
            print(f"  case {i}: API error ({exc}); marking non_conflict")
            text = ""
        parsed = parse_verdict_response(text)
        verdict = _norm_verdict(parsed.get("verdict", "")) if parsed else "non_conflict"
        if verdict not in VERDICTS:
            verdict = "non_conflict"
        predictions.append({
            "case_id": case.get("case_id", i),
            "perturbation_type": case.get("perturbation_type", ""),
            "ground_truth": case["ground_truth"],
            "predicted": verdict,
            "confidence": (parsed or {}).get("confidence", "") if parsed else "",
            "raw_response": text[:500],
        })
        if i % 10 == 0:
            print(f"  {i}/{len(cases)} done")
    elapsed = time.time() - t0

    metrics = compute_perturbation_metrics(predictions)
    out = {
        "metadata": {
            "model": model,
            "n_cases": len(cases),
            "elapsed_s": round(elapsed, 1),
            "ground_truth_distribution": gt_dist,
        },
        "metrics": metrics,
        "predictions": predictions,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[perturbation] wrote {args.out}")
    print(f"  binary  : accuracy={metrics['accuracy_binary']}  macro F1={metrics['macro_f1_binary']}")
    print(f"  3-class : accuracy={metrics['accuracy_3class']}  macro F1={metrics['macro_f1_3class']}")


# ---------------------------------------------------------------------------
# Leaderboard scorer (production matcher from holdout_leaderboard.py)
# ---------------------------------------------------------------------------

ACTION_GROUPS = [
    {"collect", "use", "process"},
    {"share", "transfer", "sell"},
    {"retain", "store"},
    {"delete"},
    {"access_right", "deletion_right", "optout_right", "portability_right"},
]
MATCH_THRESHOLD = 0.40
ACTION_REMAP = {
    "rent": "sell", "trade": "sell", "sale": "sell", "sales": "sell",
    "selling": "sell", "sold": "sell",
    "sharing": "share", "shared": "share", "shares": "share",
    "disclose": "share", "disclosed": "share", "discloses": "share",
    "disclosing": "share", "disclosure": "share", "disclosures": "share",
    "distribute": "share", "distributed": "share",
    "transmitted": "transfer", "transmit": "transfer",
    "transmission": "transfer",
    "store": "retain", "stored": "retain", "storing": "retain",
    "storage": "retain", "retention": "retain",
    "remove": "delete", "removed": "delete", "removal": "delete",
    "erase": "delete", "erased": "delete",
}


def _tokens(s: str) -> set[str]:
    if not s:
        return set()
    return {t for t in re.sub(r"[^a-z0-9]+", " ", s.lower()).split() if t}


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def action_compat(a: str, b: str) -> float:
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if a == b and a:
        return 1.0
    if not a or not b:
        return 0.0
    for grp in ACTION_GROUPS:
        if a in grp and b in grp:
            return 0.5
    return 0.0


def pair_score(g: dict, p: dict) -> float:
    return (
        0.45 * action_compat(g.get("action"), p.get("action"))
        + 0.40 * jaccard(g.get("data_object", ""), p.get("data_object", ""))
        + 0.15 * jaccard(g.get("recipient", ""), p.get("recipient", ""))
    )


def greedy_match(gold: list[dict], pred: list[dict]):
    cands = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            s = pair_score(g, p)
            if s >= MATCH_THRESHOLD:
                cands.append((s, gi, pi))
    cands.sort(reverse=True)
    used_g, used_p, matches = set(), set(), []
    for s, gi, pi in cands:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi); used_p.add(pi); matches.append((gi, pi, s))
    unmatched_g = [i for i in range(len(gold)) if i not in used_g]
    unmatched_p = [i for i in range(len(pred)) if i not in used_p]
    return matches, unmatched_g, unmatched_p


def classify_unmatched(side: str, pps: dict, other_side_pps: list[dict],
                       clause_text: str) -> tuple[str, str]:
    a = (pps.get("action") or "").strip().lower()
    if side == "pred" and a in ACTION_REMAP:
        target = ACTION_REMAP[a]
        for o in other_side_pps:
            oa = (o.get("action") or "").lower()
            if oa == target or oa == a:
                return "B", f"action {a!r} synonym of {target!r}"
    pps_do_toks = _tokens(pps.get("data_object", ""))
    clause_toks = _tokens(clause_text)
    for o in other_side_pps:
        oa = (o.get("action") or "").lower()
        if oa != a:
            continue
        o_do_toks = _tokens(o.get("data_object", ""))
        if (pps_do_toks & clause_toks) or (o_do_toks & clause_toks):
            return "A", "splitting/granularity"
    if side == "pred":
        return "D", "no compatible-action gold PPS in same clause"
    return "E", "no compatible-action pred PPS in same clause"


def cmd_leaderboard(args):
    client, model = get_client(args)
    clauses_path = Path(args.clauses) if args.clauses else DATA_DIR / "holdout_clauses_100.jsonl"
    gold_path    = Path(args.gold)    if args.gold    else DATA_DIR / "gold_claude_holdout_100_v3.jsonl"
    if not clauses_path.exists() or not gold_path.exists():
        print(f"ERROR: missing input. Need {clauses_path} and {gold_path}.", file=sys.stderr)
        sys.exit(2)

    clauses = {json.loads(l)["clause_id"]: json.loads(l) for l in clauses_path.open()}
    gold_by_clause: dict[str, list[dict]] = defaultdict(list)
    for line in gold_path.open():
        item = json.loads(line)
        cid = item.get("clause_id", "")
        if "gold_pps" in item:
            gold_by_clause[cid].extend(item["gold_pps"])
        else:
            gold_by_clause[cid].append(item)
    if args.max_cases:
        ids = list(clauses.keys())[: args.max_cases]
        clauses = {i: clauses[i] for i in ids}

    n_pred_total = n_gold_total = n_match_total = 0
    cat_pred = Counter(); cat_gold = Counter()
    per_clause = []
    t0 = time.time()
    for i, (cid, clause) in enumerate(clauses.items(), 1):
        gold_list = gold_by_clause.get(cid, [])
        prompt = EXTRACTION_PROMPT.format(
            section=clause.get("section_header", ""),
            clause=clause.get("text", ""),
        )
        try:
            text = call_chat(client, model, prompt)
        except Exception as exc:
            print(f"  clause {i}: API error ({exc}); skipping")
            text = ""
        pred_list = parse_json_array(text)
        matches, unm_g, unm_p = greedy_match(gold_list, pred_list)
        n_match_total += len(matches)
        n_pred_total  += len(pred_list)
        n_gold_total  += len(gold_list)
        clause_text = clause.get("text", "")
        for pi in unm_p:
            cat, _ = classify_unmatched("pred", pred_list[pi], gold_list, clause_text)
            cat_pred[cat] += 1
        for gi in unm_g:
            cat, _ = classify_unmatched("gold", gold_list[gi], pred_list, clause_text)
            cat_gold[cat] += 1
        per_clause.append({"clause_id": cid, "n_pred": len(pred_list),
                           "n_gold": len(gold_list), "n_match": len(matches)})
        if i % 10 == 0:
            print(f"  {i}/{len(clauses)} done (running: pred={n_pred_total} gold={n_gold_total} match={n_match_total})")
    elapsed = time.time() - t0

    p_strict = n_match_total / n_pred_total if n_pred_total else 0.0
    r_strict = n_match_total / n_gold_total if n_gold_total else 0.0
    f_strict = 2 * p_strict * r_strict / (p_strict + r_strict) if (p_strict + r_strict) else 0.0
    n_dup_pred = cat_pred["A"] + cat_pred["B"]
    n_dup_gold = cat_gold["A"]
    eff_pred = max(n_pred_total - n_dup_pred, n_match_total)
    eff_gold = max(n_gold_total - n_dup_gold, n_match_total)
    p_adj = n_match_total / eff_pred if eff_pred else 0.0
    r_adj = n_match_total / eff_gold if eff_gold else 0.0
    f_adj = 2 * p_adj * r_adj / (p_adj + r_adj) if (p_adj + r_adj) else 0.0

    out = {
        "metadata": {"model": model, "n_clauses": len(clauses),
                     "elapsed_s": round(elapsed, 1)},
        "n_pred": n_pred_total, "n_gold": n_gold_total, "n_match": n_match_total,
        "strict":   {"precision": round(p_strict, 4), "recall": round(r_strict, 4), "f1": round(f_strict, 4)},
        "adjusted": {"precision": round(p_adj,    4), "recall": round(r_adj,    4), "f1": round(f_adj,    4)},
        "cat_pred": dict(cat_pred),
        "cat_gold": dict(cat_gold),
        "per_clause": per_clause,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[leaderboard] wrote {args.out}")
    print(f"  strict  : P={out['strict']['precision']} R={out['strict']['recall']} F1={out['strict']['f1']}")
    print(f"  adjusted: P={out['adjusted']['precision']} R={out['adjusted']['recall']} F1={out['adjusted']['f1']}")


# ---------------------------------------------------------------------------
# Verdict agreement (per-model column)
# ---------------------------------------------------------------------------

def build_finding_prompt(f: dict) -> str:
    s1 = {
        "actor": f.get("statement_1_actor", ""),
        "action": f.get("statement_1_action", ""),
        "modality": f.get("statement_1_modality", ""),
        "data_object": f.get("statement_1_data_object", ""),
        "purpose": f.get("statement_1_purpose", ""),
        "condition": f.get("statement_1_condition", ""),
        "source_text": f.get("statement_1_text", ""),
        "policy_source": "first_party",
    }
    s2 = {
        "actor": f.get("statement_2_actor", ""),
        "action": f.get("statement_2_action", ""),
        "modality": f.get("statement_2_modality", ""),
        "data_object": f.get("statement_2_data_object", ""),
        "purpose": f.get("statement_2_purpose", ""),
        "condition": f.get("statement_2_condition", ""),
        "source_text": f.get("statement_2_text", ""),
        "policy_source": f"third_party:{f.get('vendor_name', 'vendor')}",
    }
    return build_verifier_prompt(s1, s2, pattern_id=f.get("pattern_id", ""))


def cmd_agreement(args):
    client, model = get_client(args)
    findings_path = Path(args.findings) if args.findings else DATA_DIR / "verdict_agreement_findings.jsonl"
    if not findings_path.exists():
        print(f"ERROR: missing {findings_path}. Extract data/dataset.tar.gz first.", file=sys.stderr)
        sys.exit(2)
    findings = [json.loads(l) for l in findings_path.open()]
    if args.max_cases:
        findings = findings[: args.max_cases]

    verdicts = []
    t0 = time.time()
    for i, f in enumerate(findings, 1):
        prompt = build_finding_prompt(f)
        try:
            text = call_chat(client, model, prompt)
        except Exception as exc:
            print(f"  finding {i}: API error ({exc}); marking non_conflict")
            text = ""
        parsed = parse_verdict_response(text)
        v = _norm_verdict(parsed.get("verdict", "")) if parsed else "non_conflict"
        if v not in VERDICTS:
            v = "non_conflict"
        verdicts.append({
            "inconsistency_id": f.get("inconsistency_id", ""),
            "pair_id": f.get("pair_id", ""),
            "predicted": v,
            "pattern_verdict": _norm_verdict(f.get("pattern_verdict", "")),
            "confidence": (parsed or {}).get("confidence", "") if parsed else "",
            "raw_response": text[:500],
        })
        if i % 10 == 0:
            print(f"  {i}/{len(findings)} done — running counts: {dict(Counter(v['predicted'] for v in verdicts))}")
    elapsed = time.time() - t0

    out = {
        "metadata": {"model": model, "n_findings": len(findings),
                     "elapsed_s": round(elapsed, 1),
                     "verdict_distribution": dict(Counter(v["predicted"] for v in verdicts))},
        "verdicts": verdicts,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[agreement] wrote {args.out}")
    print(f"  distribution: {out['metadata']['verdict_distribution']}")
    print(f"  Run this for several served models, then aggregate the verdicts.json files "
          f"(matching by inconsistency_id) to compute pairwise agreement.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_common_flags(p):
    p.add_argument("--base-url", default=None, help="OpenAI-compatible /v1 URL (default: env LLM_BASE_URL)")
    p.add_argument("--api-key",  default=None, help="API key (default: env LLM_API_KEY)")
    p.add_argument("--model",    default=None, help="Model name (default: env LLM_MODEL)")
    p.add_argument("--max-cases", type=int, default=None, help="Cap input size (smoke test)")
    p.add_argument("--out", required=True, help="Output JSON path")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pert = sub.add_parser("perturbation", help="Verifier perturbation eval (Evaluation §4.2.1)")
    add_common_flags(p_pert)
    p_pert.add_argument("--cases", default=None, help="Override input JSONL")
    p_pert.set_defaults(func=cmd_perturbation)

    p_lb = sub.add_parser("leaderboard", help="Extractor leaderboard eval (Evaluation §4.1)")
    add_common_flags(p_lb)
    p_lb.add_argument("--clauses", default=None, help="Override clauses JSONL")
    p_lb.add_argument("--gold",    default=None, help="Override gold-PPS JSONL")
    p_lb.set_defaults(func=cmd_leaderboard)

    p_agr = sub.add_parser("agreement", help="Verdict-agreement eval (Evaluation §4.2.2)")
    add_common_flags(p_agr)
    p_agr.add_argument("--findings", default=None, help="Override findings JSONL")
    p_agr.set_defaults(func=cmd_agreement)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
