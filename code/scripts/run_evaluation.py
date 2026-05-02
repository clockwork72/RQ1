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
    (`VERIFIER_PROMPT_LEGACY` from `code/prompts/unified_prompts.py`) for a
    verdict on each, and computes 3-class precision / recall / F1 plus the
    binary view used in the paper.

leaderboard
    Re-runs one row of the extractor leaderboard (Evaluation §4.1) on the
    100-clause holdout in `data/raw/benchmarks/holdout_clauses_100.jsonl`,
    matches the predicted PPSes against
    `data/raw/benchmarks/gold_claude_holdout_100_v3.jsonl` with a bipartite
    matcher, and writes strict + adjusted P / R / F1.

agreement
    Re-runs one model's column of the verdict-agreement eval (Evaluation
    §4.2.2). Reads the 100 inconsistent findings sampled with seed 42
    from `data/raw/benchmarks/verdict_agreement_findings.jsonl`, asks the
    verifier (`VERIFIER_PROMPT_LEGACY`) for a verdict on each, and writes
    a per-model verdicts JSON. Re-running this with a different served
    model gives you the corresponding column of the agreement table; use
    several columns together to reproduce the pairwise-agreement numbers.

Common flags
------------
--base-url / env LLM_BASE_URL   default http://localhost:8000/v1
--api-key  / env LLM_API_KEY    default "local"
--model    / env LLM_MODEL      default gemma3:27b
--out                           output path (per-task default below)
--max-cases                     cap for smoke tests; default = full set

Quick start
-----------

    bash code/llm_serving/serve_vllm.sh                 # start a local 2x A100 server
    source code/llm_serving/env_local.sh                # exports LLM_BASE_URL etc.

    python code/scripts/run_evaluation.py perturbation \\
        --out out/eval_perturbation.json
    python code/scripts/run_evaluation.py leaderboard \\
        --out out/eval_leaderboard.json
    python code/scripts/run_evaluation.py agreement \\
        --out out/eval_agreement.json

Each subcommand prints a one-line summary at the end so you can compare
against the paper's numbers without opening the JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai (>=1.0)", file=sys.stderr)
    sys.exit(2)

from prompts.unified_prompts import (
    EXTRACTION_PROMPT,
    REFLECTION_EXHAUSTION_PROMPT,
    REFLECTION_RECOVERY_PROMPT,
    VERIFIER_PROMPT_LEGACY,
)

VERDICTS = ["inconsistent", "underspecified", "non_conflict"]
DATA_DIR = REPO_ROOT / "data" / "raw" / "benchmarks"


def get_client(args) -> tuple[OpenAI, str]:
    base_url = args.base_url or os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
    api_key  = args.api_key  or os.environ.get("LLM_API_KEY", "local")
    model    = args.model    or os.environ.get("LLM_MODEL", "gemma3:27b")
    print(f"[run_evaluation] endpoint = {base_url}")
    print(f"[run_evaluation] model    = {model}")
    return OpenAI(base_url=base_url, api_key=api_key, timeout=120), model


def parse_verdict(text: str) -> str:
    """Extract the verdict label from an LLM response."""
    if not text:
        return "non_conflict"
    m = re.search(r'"verdict"\s*:\s*"([a-z_]+)"', text)
    if m and m.group(1) in VERDICTS:
        return m.group(1)
    low = text.lower()
    for v in VERDICTS:
        if v in low:
            return v
    return "non_conflict"


def call_chat(client: OpenAI, model: str, prompt: str, temperature: float = 0.0) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=2048,
    )
    return (resp.choices[0].message.content or "").strip()


def parse_json_array(text: str) -> list[dict]:
    if not text:
        return []
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.MULTILINE)
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


def build_verifier_prompt(case: dict) -> str:
    s1 = case.get("statement_1", {})
    s2 = case.get("statement_2", {})

    def field(stmt, key, default=""):
        val = stmt.get(key, default)
        return str(val) if val is not None else default

    return VERIFIER_PROMPT_LEGACY.format(
        pattern_id=case.get("pattern_id", ""),
        pattern_name=case.get("pattern_name", ""),
        source_text_1=field(s1, "source_text") or field(s1, "text"),
        action_1=field(s1, "action"),
        modality_1=field(s1, "modality"),
        data_1=field(s1, "data_object"),
        source_text_2=field(s2, "source_text") or field(s2, "text"),
        action_2=field(s2, "action"),
        modality_2=field(s2, "modality"),
        data_2=field(s2, "data_object"),
        pattern_explanation=case.get("explanation", ""),
    )


def build_finding_prompt(finding: dict) -> str:
    return VERIFIER_PROMPT_LEGACY.format(
        pattern_id=finding.get("pattern_id", ""),
        pattern_name=finding.get("pattern_name", ""),
        source_text_1=finding.get("statement_1_text", ""),
        action_1=finding.get("statement_1_action", ""),
        modality_1=finding.get("statement_1_modality", ""),
        data_1=finding.get("statement_1_data_object", ""),
        source_text_2=finding.get("statement_2_text", ""),
        action_2=finding.get("statement_2_action", ""),
        modality_2=finding.get("statement_2_modality", ""),
        data_2=finding.get("statement_2_data_object", ""),
        pattern_explanation="",
    )


def metrics_3class(predictions: list[dict]) -> dict:
    """Per-class precision / recall / F1 + macro / micro accuracy."""
    n = len(predictions)
    correct = sum(1 for p in predictions if p["predicted"] == p["ground_truth"])
    overall_acc = correct / n if n else 0.0
    per_class = {}
    for cls in VERDICTS:
        tp = sum(1 for p in predictions if p["predicted"] == cls and p["ground_truth"] == cls)
        fp = sum(1 for p in predictions if p["predicted"] == cls and p["ground_truth"] != cls)
        fn = sum(1 for p in predictions if p["predicted"] != cls and p["ground_truth"] == cls)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[cls] = {"precision": round(precision, 4), "recall": round(recall, 4),
                          "f1": round(f1, 4), "support": tp + fn}
    macro_f1 = sum(c["f1"] for c in per_class.values()) / len(per_class)
    return {
        "overall_accuracy": round(overall_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
    }


def metrics_binary(predictions: list[dict]) -> dict:
    """Binary view: positive = (inconsistent or underspecified), negative = non_conflict."""
    def to_bin(v): return "non_conflict" if v == "non_conflict" else "genuine"
    tp = sum(1 for p in predictions if to_bin(p["predicted"]) == "genuine" and to_bin(p["ground_truth"]) == "genuine")
    fp = sum(1 for p in predictions if to_bin(p["predicted"]) == "genuine" and to_bin(p["ground_truth"]) == "non_conflict")
    fn = sum(1 for p in predictions if to_bin(p["predicted"]) == "non_conflict" and to_bin(p["ground_truth"]) == "genuine")
    tn = sum(1 for p in predictions if to_bin(p["predicted"]) == "non_conflict" and to_bin(p["ground_truth"]) == "non_conflict")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def cmd_perturbation(args):
    client, model = get_client(args)
    cases_path = Path(args.cases) if args.cases else DATA_DIR / "perturbation_cases.jsonl"
    if not cases_path.exists():
        print(f"ERROR: missing {cases_path}. Extract data/dataset.tar.gz first "
              f"(or run any notebook once).", file=sys.stderr)
        sys.exit(2)
    cases = [json.loads(l) for l in cases_path.open()]
    if args.max_cases:
        cases = cases[: args.max_cases]
    print(f"[perturbation] {len(cases)} cases (gt: {dict(Counter(c['ground_truth'] for c in cases))})")

    predictions = []
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        prompt = build_verifier_prompt(case)
        try:
            text = call_chat(client, model, prompt)
        except Exception as exc:
            print(f"  case {i}: API error ({exc}); marking non_conflict")
            text = ""
        verdict = parse_verdict(text)
        predictions.append({
            "case_id": case.get("case_id", i),
            "ground_truth": case["ground_truth"],
            "predicted": verdict,
            "raw_response": text[:500],
        })
        if i % 10 == 0:
            print(f"  {i}/{len(cases)} done")
    elapsed = time.time() - t0

    out = {
        "metadata": {
            "model": model,
            "n_cases": len(cases),
            "elapsed_s": round(elapsed, 1),
            "ground_truth_distribution": dict(Counter(c["ground_truth"] for c in cases)),
        },
        "metrics_3class": metrics_3class(predictions),
        "metrics_binary": metrics_binary(predictions),
        "predictions": predictions,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[perturbation] wrote {args.out}")
    print(f"  3-class macro F1 : {out['metrics_3class']['macro_f1']}")
    print(f"  binary F1        : {out['metrics_binary']['f1']}  (P={out['metrics_binary']['precision']}, R={out['metrics_binary']['recall']})")


def call_extractor(client: OpenAI, model: str, clause_text: str, section: str = "") -> list[dict]:
    """One round of extraction; returns the raw items list."""
    prompt = EXTRACTION_PROMPT.format(section=section, clause=clause_text)
    text = call_chat(client, model, prompt)
    return parse_json_array(text)


def normalize_pps(item: dict) -> tuple:
    """Tuple key for matching: (action, data_object, modality, condition)."""
    return (
        str(item.get("action", "")).lower().strip(),
        str(item.get("data_object", "")).lower().strip(),
        str(item.get("modality", "")).upper().strip(),
        str(item.get("condition", "")).lower().strip(),
    )


def adjusted_match(pred: dict, gold: dict) -> bool:
    """Loose match: action + data_object equal (modality/condition may diverge)."""
    return (
        str(pred.get("action", "")).lower().strip() == str(gold.get("action", "")).lower().strip()
        and str(pred.get("data_object", "")).lower().strip() == str(gold.get("data_object", "")).lower().strip()
    )


def score_clause(pred_pps: list[dict], gold_pps: list[dict]) -> dict:
    used_strict, used_adj = set(), set()
    tp_strict = tp_adj = 0
    for p in pred_pps:
        pkey = normalize_pps(p)
        for j, g in enumerate(gold_pps):
            if j in used_strict:
                continue
            if pkey == normalize_pps(g):
                used_strict.add(j); tp_strict += 1; break
        for j, g in enumerate(gold_pps):
            if j in used_adj:
                continue
            if adjusted_match(p, g):
                used_adj.add(j); tp_adj += 1; break
    return {
        "n_pred": len(pred_pps),
        "n_gold": len(gold_pps),
        "tp_strict": tp_strict,
        "tp_adjusted": tp_adj,
    }


def cmd_leaderboard(args):
    client, model = get_client(args)
    clauses_path = Path(args.clauses) if args.clauses else DATA_DIR / "holdout_clauses_100.jsonl"
    gold_path    = Path(args.gold)    if args.gold    else DATA_DIR / "gold_claude_holdout_100_v3.jsonl"
    if not clauses_path.exists() or not gold_path.exists():
        print(f"ERROR: missing input. Need {clauses_path} and {gold_path}. "
              f"Extract data/dataset.tar.gz first.", file=sys.stderr)
        sys.exit(2)

    clauses = [json.loads(l) for l in clauses_path.open()]
    gold_by_clause: dict[str, list[dict]] = {}
    for line in gold_path.open():
        item = json.loads(line)
        gold_by_clause.setdefault(item.get("clause_id", ""), []).append(item)
    if args.max_cases:
        clauses = clauses[: args.max_cases]

    n_clauses = 0
    sum_pred = sum_gold = sum_tp_strict = sum_tp_adj = 0
    per_clause = []
    t0 = time.time()
    for i, clause in enumerate(clauses, 1):
        cid = clause.get("clause_id", "")
        gold = gold_by_clause.get(cid, [])
        try:
            pred = call_extractor(client, model, clause["text"], clause.get("section_header", ""))
        except Exception as exc:
            print(f"  clause {i}: API error ({exc}); skipping")
            pred = []
        scores = score_clause(pred, gold)
        sum_pred += scores["n_pred"]
        sum_gold += scores["n_gold"]
        sum_tp_strict += scores["tp_strict"]
        sum_tp_adj    += scores["tp_adjusted"]
        n_clauses += 1
        per_clause.append({"clause_id": cid, **scores})
        if i % 10 == 0:
            print(f"  {i}/{len(clauses)} done (so far: pred={sum_pred} gold={sum_gold} tp_strict={sum_tp_strict} tp_adj={sum_tp_adj})")
    elapsed = time.time() - t0

    p_strict = sum_tp_strict / sum_pred if sum_pred else 0.0
    r_strict = sum_tp_strict / sum_gold if sum_gold else 0.0
    f_strict = 2 * p_strict * r_strict / (p_strict + r_strict) if (p_strict + r_strict) else 0.0
    p_adj = sum_tp_adj / sum_pred if sum_pred else 0.0
    r_adj = sum_tp_adj / sum_gold if sum_gold else 0.0
    f_adj = 2 * p_adj * r_adj / (p_adj + r_adj) if (p_adj + r_adj) else 0.0

    out = {
        "metadata": {
            "model": model,
            "n_clauses": n_clauses,
            "n_pred": sum_pred,
            "n_gold": sum_gold,
            "elapsed_s": round(elapsed, 1),
        },
        "strict":   {"precision": round(p_strict, 4), "recall": round(r_strict, 4), "f1": round(f_strict, 4)},
        "adjusted": {"precision": round(p_adj,    4), "recall": round(r_adj,    4), "f1": round(f_adj,    4)},
        "per_clause": per_clause,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[leaderboard] wrote {args.out}")
    print(f"  strict  : P={out['strict']['precision']} R={out['strict']['recall']} F1={out['strict']['f1']}")
    print(f"  adjusted: P={out['adjusted']['precision']} R={out['adjusted']['recall']} F1={out['adjusted']['f1']}")


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
        v = parse_verdict(text)
        verdicts.append({
            "inconsistency_id": f.get("inconsistency_id", ""),
            "predicted": v,
            "pattern_verdict": f.get("pattern_verdict", ""),
            "raw_response": text[:500],
        })
        if i % 10 == 0:
            print(f"  {i}/{len(findings)} done — running counts: {dict(Counter(v['predicted'] for v in verdicts))}")
    elapsed = time.time() - t0

    out = {
        "metadata": {
            "model": model,
            "n_findings": len(findings),
            "elapsed_s": round(elapsed, 1),
            "verdict_distribution": dict(Counter(v["predicted"] for v in verdicts)),
        },
        "verdicts": verdicts,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[agreement] wrote {args.out}")
    print(f"  distribution: {out['metadata']['verdict_distribution']}")
    print(f"  Run this for several served models, then aggregate the verdicts.json files "
          f"(matching by inconsistency_id) to compute pairwise agreement.")


def add_common_flags(p):
    p.add_argument("--base-url", default=None, help="OpenAI-compatible /v1 URL (default: env LLM_BASE_URL)")
    p.add_argument("--api-key",  default=None, help="API key (default: env LLM_API_KEY)")
    p.add_argument("--model",    default=None, help="Model name to send to the endpoint (default: env LLM_MODEL)")
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
