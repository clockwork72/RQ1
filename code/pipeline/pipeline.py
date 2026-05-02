"""End-to-end pipeline orchestration."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

from .schema import Verdict
from .config import (
    OUTPUT_DIR, EXTRACTION_BACKEND, EXTRACTION_PROMPT_VERSION,
    EXTRACTION_REFLECTION_ENABLED, EXTRACTION_REFLECTION_ROUNDS,
    EXTRACTION_TEMPERATURE, LLAMACPP_BASE_URL, LLAMACPP_MODEL_NAME,
    SCOPE_PROMPT_VERSION, VERIFIER_PROMPT_VERSION,
    CLUSTER_VERIFY_PROMPT_VERSION, BUILD_PPS_VERSION,
)
from .extractor import (
    attribute_finding_to_gdpr,
    compare_gdpr_completeness,
    compute_clause_gdpr_coverage,
    compute_extraction_gap,
    compute_gdpr_completeness,
    extract_pps_from_policy,
    segment_clauses,
)
from graph import build_graph, compute_graph_metrics, merge_graphs
from .normalizer import DATA_ONTOLOGY, DATA_SYNONYMS, PURPOSE_SYNONYMS
from .pair_cache import load_pair_cache, pair_cache_key, save_pair_cache
from .patterns import PATTERN_IDS, run_all_patterns
from .verifier import verify_candidates
from visualize import visualize_graph

# Optional per-stage debug dumping — no-op unless PIPELINE_DEBUG=1 is set.
try:
    from bert_extraction.e2e_audit.scripts import debug_dump as _dd
except Exception:  # pragma: no cover - debug is optional
    _dd = None

# ─────────────────────────────────────────────────────────────────────────
# Reproducibility metadata — captured in every pair's results JSON so a
# reviewer can reconstruct which commit, which ontology hash, and which
# pattern set produced a given output. Cached at module load.
# ─────────────────────────────────────────────────────────────────────────

def _git_commit_sha() -> str:
    """Return current HEAD commit SHA (short). Returns '' outside a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=3,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _normalizer_ontology_hash() -> str:
    """Hash the ontology + synonym tables so downstream can detect drift.
    A canonicalization change invalidates graph equivalence."""
    payload = json.dumps({
        "data_ontology": {k: sorted(v) for k, v in DATA_ONTOLOGY.items()},
        "data_synonyms": dict(sorted(DATA_SYNONYMS.items())),
        "purpose_synonyms": dict(sorted(PURPOSE_SYNONYMS.items())),
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _patterns_version() -> str:
    """String identity of the active pattern set. Changes when patterns are
    added/removed — lets reviewers tell whether a results JSON came from a
    Round 9/10/11 pattern lineup."""
    return "+".join(sorted(PATTERN_IDS.values()))


# Compute once at import time so repro metadata is deterministic per run.
_COMMIT_SHA = _git_commit_sha()
_ONTOLOGY_HASH = _normalizer_ontology_hash()
_PATTERNS_VERSION = _patterns_version()


BROAD_DATA_OBJECTS = {
    "personal data",
    "contact information",
    "device information",
    "behavioral data",
    "financial data",
    "health data",
    "biometric data",
    "geolocation",
}


DEMO_WEBSITE_POLICY = """
Privacy Policy - ExampleShop.com

Last Updated: January 15, 2025

1. Information We Collect

We collect the following personal information when you create an account or make a purchase:
- Your name and email address
- Your postal address for shipping
- Payment information to process transactions
- Phone number for order notifications

We also collect browsing history and IP address when you visit our website, with your consent, to improve your shopping experience.

2. How We Use Your Information

We use your personal data for the following purposes:
- To provide and deliver our services
- To process your orders and payments
- To send you order confirmations and shipping updates
- For analytics purposes, to understand how our website is used

We do not use your personal information for targeted advertising. We do not engage in automated decision-making or profiling based on your personal data.

3. Third-Party Sharing

We share your browsing history and device information with our analytics provider, Google Analytics, to understand website traffic patterns. This sharing requires your explicit consent through our cookie consent banner.

We do not sell your personal data to any third party. We will never share your email address or payment information with advertising companies.

We may share your information with law enforcement when required by law.

4. Data Retention

We retain your name and email address for the duration of your account. Your browsing history is retained for 90 days and then automatically deleted. Payment information is retained for 30 days after the transaction is complete. IP addresses are stored for 14 days for security purposes.

5. Your Rights

You have the right to access all personal data we hold about you. You can request deletion of your personal data at any time by contacting our Data Protection Officer. You have the right to data portability - we will provide your data in a machine-readable format upon request. You may opt out of analytics tracking at any time through your account settings.

6. Consent

We collect and process your browsing history only with your explicit consent, obtained through our cookie consent mechanism. You may withdraw your consent at any time.

7. International Transfers

Your personal data may be transferred to and processed in countries outside the European Economic Area. We ensure appropriate safeguards are in place for all international transfers.

8. Children's Data

We do not knowingly collect personal data from children under 16 years of age. If we discover we have collected data from a child, we will delete it immediately.

9. Security

We implement industry-standard security measures to protect your personal data, including encryption of data in transit and at rest.

10. Contact

For any privacy-related inquiries, contact our Data Protection Officer at dpo@exampleshop.com.
"""


DEMO_VENDOR_POLICY = """
Google Analytics - Data Processing Terms and Privacy Disclosure

Effective Date: March 1, 2025

1. Data We Collect

Google Analytics automatically collects the following data from websites that use our service:
- Browsing history and page views
- Cookie IDs and advertising IDs
- IP address and device information (browser type, operating system, device type)
- Search queries performed on the website
- Clickstream data and user interactions

This data is collected by default when the Google Analytics tracking code is installed on a website. No additional user consent is required for basic data collection by Google Analytics.

2. How We Use Data

We use the collected data for the following purposes:
- Analytics and reporting for the website operator
- Product improvement of Google services
- Advertising and ad personalization across Google's advertising network
- Research and development of new features

We may use browsing history and cookie IDs for targeted advertising purposes. Device information and IP addresses are used for analytics and fraud prevention.

3. Data Sharing

We share aggregated analytics data with the website operator. We also share user-level data, including browsing history and advertising IDs, with our advertising partners for ad personalization.

Google may share data with third-party advertising networks that participate in our advertising ecosystem.

4. Data Retention

Browsing history and user interaction data is retained for 26 months by default. Website operators can configure retention to 14 or 38 months. Cookie IDs are retained for 24 months. IP addresses are retained for 9 months.

Aggregated and anonymized data may be retained indefinitely for trend analysis and research purposes.

5. User Controls and Rights

Users may opt out of Google Analytics tracking by installing the Google Analytics Opt-out Browser Add-on. Users can manage ad personalization settings through Google's Ad Settings page.

Google does not provide a direct mechanism for individual users to request deletion of their analytics data. Data deletion requests should be directed to the website operator, who can use the Google Analytics Data Deletion API.

6. International Transfers

Data collected by Google Analytics may be transferred to and processed in the United States and other countries where Google operates data centers.

7. Automated Decision-Making

Google Analytics data may be used in automated systems for ad targeting and content personalization. These systems use machine learning to create user segments and predict user interests.

8. Updates

We may update these terms at any time. Continued use of Google Analytics after updates constitutes acceptance of the revised terms.
"""


def _summary_counts(inconsistencies: list) -> dict[str, dict[str, int]]:
    summary = {
        "by_severity": {name: 0 for name in ("CRITICAL", "HIGH", "MEDIUM", "LOW")},
        "by_verdict": {
            name: 0 for name in ("inconsistent", "underspecified", "non_conflict")
        },
        "by_pattern": {},
    }
    for inconsistency in inconsistencies:
        severity = (
            inconsistency["severity"]
            if isinstance(inconsistency, dict)
            else inconsistency.severity.value
        )
        verdict = (
            inconsistency["verdict"]
            if isinstance(inconsistency, dict)
            else inconsistency.verdict.value
        )
        pattern_id = (
            inconsistency["pattern_id"]
            if isinstance(inconsistency, dict)
            else inconsistency.pattern_id
        )
        summary["by_severity"][severity] += 1
        summary["by_verdict"][verdict] += 1
        summary["by_pattern"][pattern_id] = (
            summary["by_pattern"].get(pattern_id, 0) + 1
        )
    return summary


def _severity_rank(value: str) -> int:
    order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    return order.get(value, 9)


def _pattern_rank(value: str) -> int:
    order = {"Π₈": 0, "Π₁": 1, "Π₄": 2, "Π₅": 3, "Π₂": 4}
    return order.get(value, 9)


def _is_cross_policy_inconsistency(item: dict) -> bool:
    left = item["statement_1"].get("policy_source", "")
    right = item["statement_2"].get("policy_source", "")
    return left != right


def _statement_quality_score(statement: dict) -> int:
    score = 0
    data_object = (statement.get("data_object") or "").strip().lower()
    purpose = (statement.get("purpose") or "").strip().lower()
    recipient = (statement.get("recipient") or "").strip()
    condition = statement.get("condition") or "unspecified"
    temporality = statement.get("temporality") or "unspecified"
    source_text = statement.get("source_text") or ""

    if data_object and data_object not in BROAD_DATA_OBJECTS:
        score += 3
    elif data_object:
        score += 1
    if purpose:
        score += 2
    if recipient:
        score += 1
    if condition != "unspecified":
        score += 1
    if temporality != "unspecified":
        score += 1
    if statement.get("is_negative"):
        score += 2

    source_length = len(source_text.strip())
    if source_length and source_length <= 280:
        score += 2
    elif source_length <= 500:
        score += 1
    elif source_length >= 900:
        score -= 2
    elif source_length >= 650:
        score -= 1

    if "|" in source_text:
        score -= 2
    lowered = source_text.lower()
    if "controller" in lowered and "responsible for" in lowered:
        score -= 2
    return score


# Per-pattern cap applied in both the curated-output stage and the research
# annotation-CSV stage. Until 2026-04-18 the two code paths drifted
# (curate used {Π₁:20, Π₂:15, Π₄:20, Π₅:15}; research used {Π₁:12, Π₂:12,
# Π₄:25, Π₅:12}). Unified here to a single source of truth. The
# research-CSV stage still gets a relaxed 1.5× backfill pass via
# `relaxed_pattern_caps` for sparse pairs.
PATTERN_FINDING_CAPS = {"Π₈": 30, "Π₁": 20, "Π₂": 15, "Π₄": 20, "Π₅": 15}


def _inconsistency_priority(item: dict) -> tuple[int, int, int, int, int, str]:
    """Sort key for finding curation. Higher element values sort earlier.

    Tuple layout (P2 added element 1):
      0. cross_policy           1 if Π₈ cross-policy, 0 else
      1. bridge_confirmed       1 if the finding has a confirmed bridge-flow
                                 (the strongest evidence available); 0 else
      2. severity_score         CRITICAL=4 / HIGH=3 / MEDIUM=2 / LOW=1
      3. pattern_score          Π₈=8 / Π₁=5 / Π₂/Π₄/Π₅=4
      4. verdict+evidence       finding-quality proxy
      5. inconsistency_id       tie-breaker (stable sort)
    """
    cross_policy = _is_cross_policy_inconsistency(item)
    # P2: bridge-confirmed findings — the Π₈ pair has an actual
    # SENDS_DATA_VIA → bridge → DATA_PROCESSED_BY chain verified in the
    # merged graph. This is the strongest evidence signal we have and
    # should bubble to the top of the curated list.
    nh_ctx = item.get("neighborhood_context") or {}
    bridge_confirmed = bool(nh_ctx.get("bridge_confirmed", False))

    severity_score = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(
        item["severity"], 0
    )
    verdict_score = {
        "inconsistent": 3,
        "underspecified": 2,
        "non_conflict": 1,
    }.get(item["verdict"], 0)
    pattern_score = {"Π₈": 8, "Π₁": 5, "Π₄": 4, "Π₅": 4, "Π₂": 4}.get(
        item["pattern_id"], 0
    )
    evidence_score = _statement_quality_score(item["statement_1"]) + _statement_quality_score(
        item["statement_2"]
    )

    explanation = item.get("explanation", "").lower()
    if "advertising" in explanation or "targeted advertising" in explanation:
        evidence_score += 3
    if "retains indefinitely" in explanation or "retains data for" in explanation:
        evidence_score += 2
    if "no corresponding mechanism" in explanation:
        evidence_score += 3
    if "requires '" in explanation and "operates '" in explanation:
        evidence_score += 2

    return (
        1 if cross_policy else 0,
        1 if bridge_confirmed else 0,
        severity_score,
        pattern_score,
        verdict_score + evidence_score,
        item["inconsistency_id"],
    )


def _curation_signature(item: dict) -> tuple[str, ...]:
    left = item["statement_1"]
    right = item["statement_2"]
    if _is_cross_policy_inconsistency(item):
        return (
            item["pattern_id"],
            left.get("policy_source", ""),
            right.get("policy_source", ""),
            left.get("source_text", "").strip(),
            right.get("source_text", "").strip(),
            item.get("explanation", ""),
        )
    return (
        item["pattern_id"],
        left.get("action", ""),
        left.get("data_object", ""),
        right.get("data_object", ""),
    )


def _curate_inconsistencies(inconsistencies: list[dict], max_total: int = 120) -> list[dict]:
    per_pattern_caps = PATTERN_FINDING_CAPS
    ordered = sorted(
        inconsistencies,
        key=lambda item: (
            -_inconsistency_priority(item)[0],
            -_inconsistency_priority(item)[1],
            -_inconsistency_priority(item)[2],
            -_inconsistency_priority(item)[3],
            -_inconsistency_priority(item)[4],
            item["inconsistency_id"],
        ),
    )

    selected: list[dict] = []
    per_pattern_counts: dict[str, int] = {}
    per_statement_counts: dict[str, int] = {}
    seen_signatures: set[tuple[str, ...]] = set()
    available_cross_policy = sum(1 for item in ordered if _is_cross_policy_inconsistency(item))
    cross_policy_target = min(max_total // 3, 12, available_cross_policy)

    for item in ordered:
        pattern_id = item["pattern_id"]
        same_policy = (
            item["statement_1"].get("policy_source", "")
            == item["statement_2"].get("policy_source", "")
        )
        same_source_clause = (
            item["statement_1"].get("source_text", "").strip()
            and item["statement_1"].get("source_text", "").strip()
            == item["statement_2"].get("source_text", "").strip()
        )
        if same_policy and same_source_clause and pattern_id != "Π₁":
            continue
        left = item["statement_1"]["id"]
        right = item["statement_2"]["id"]
        signature = _curation_signature(item)
        if signature in seen_signatures:
            continue
        if per_pattern_counts.get(pattern_id, 0) >= per_pattern_caps.get(pattern_id, 15):
            continue
        statement_limit = 4 if _is_cross_policy_inconsistency(item) else 2
        if per_statement_counts.get(left, 0) >= statement_limit or per_statement_counts.get(right, 0) >= statement_limit:
            continue
        if len(selected) < cross_policy_target and not _is_cross_policy_inconsistency(item):
            continue

        selected.append(item)
        seen_signatures.add(signature)
        per_pattern_counts[pattern_id] = per_pattern_counts.get(pattern_id, 0) + 1
        per_statement_counts[left] = per_statement_counts.get(left, 0) + 1
        per_statement_counts[right] = per_statement_counts.get(right, 0) + 1
        if len(selected) >= max_total:
            break

    return selected


def _derive_first_party_names(pair_id: str, policy_text: str = "") -> list[str]:
    """Derive likely first-party company names from the pair_id and policy text.

    Extracts the domain base name from pair_id, and also scans the first few lines
    of the policy text for "Privacy Policy - CompanyName" patterns.
    Works generically for any scraped website — no pair-specific hardcoding.
    """
    names: list[str] = []
    if "__" in pair_id:
        website_domain = pair_id.split("__")[0]
        domain = website_domain.replace("_", ".")
        base = domain.split(".")[0].capitalize()
        names.extend([base, domain])

    # Try to extract company name from policy header (first 500 chars)
    if policy_text:
        header = policy_text[:500]
        for pattern in [
            r"Privacy Policy\s*[-–—:]\s*(.+?)(?:\n|$)",
            r"^(.+?)\s+Privacy Policy",
        ]:
            match = re.search(pattern, header, re.IGNORECASE | re.MULTILINE)
            if match:
                candidate = match.group(1).strip().rstrip(".")
                if 3 <= len(candidate) <= 60:
                    names.append(candidate)
                break

    return names


def run_pair(
    website_text: str,
    vendor_text: str,
    vendor_name: str,
    service_type: str = "analytics",
    output_dir: str | Path | None = None,
    pair_id: str = "pair",
) -> dict:
    """Process one website/vendor pair end to end."""

    output_path = Path(output_dir) if output_dir else OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    # Debug dumping (only when PIPELINE_DEBUG=1). stage_path is None when off.
    stage_path = _dd.stages_dir(output_path, pair_id) if _dd else None
    if _dd:
        _dd.set_current_pair(pair_id)
        _dd.reset_trace(pair_id)

    import time as _time
    _step_timings: dict[str, float] = {}
    _pair_t0 = _time.monotonic()

    def _step_start(name: str):
        _step_timings[f"{name}_start"] = _time.monotonic()

    def _step_end(name: str):
        elapsed = _time.monotonic() - _step_timings[f"{name}_start"]
        _step_timings[name] = elapsed

    print(f"\n{'#' * 70}")
    print(f"# Processing pair: {pair_id}")
    print(f"# Vendor: {vendor_name} ({service_type})")
    print(f"{'#' * 70}")

    # Pair-level cache — disk-backed, content-addressed on (policy texts +
    # vendor metadata + pinned code versions). A hit means re-running every
    # downstream stage would produce the same output, so we splice in the
    # current pair_id/timestamp and skip straight to the per-pair JSON
    # write.
    _pcache_key = pair_cache_key(
        website_text, vendor_text, vendor_name, service_type,
        _PATTERNS_VERSION,
    )
    _pcache_hit = load_pair_cache(_pcache_key)
    if _pcache_hit is not None:
        print(f"  [pair-cache HIT] key {_pcache_key[:16]}… — reusing prior analysis")
        result = json.loads(json.dumps(_pcache_hit))  # deep copy
        result["pair_id"] = pair_id
        result["timestamp"] = datetime.now().isoformat()
        result["elapsed_s"] = round(_time.monotonic() - _pair_t0, 2)
        result.setdefault("pair_cache", {})
        result["pair_cache"].update({"hit": True, "key": _pcache_key})

        json_path = output_path / f"{pair_id}_results.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"  Results JSON: {json_path}")

        gdpr_path = output_path / f"{pair_id}_gdpr_completeness.json"
        with gdpr_path.open("w", encoding="utf-8") as handle:
            json.dump({
                "pair_id": pair_id,
                "vendor_name": vendor_name,
                "timestamp": result["timestamp"],
                "gdpr_completeness": result.get("gdpr_completeness"),
                "gdpr_comparison": result.get("gdpr_comparison"),
            }, handle, indent=2)
        print(f"  GDPR completeness JSON: {gdpr_path}")
        return result

    _step_start("extract")
    print("\n[Step 1] Extracting Privacy Practice Statements...")
    # Debug: dump segmented clauses before extraction.
    if stage_path is not None:
        try:
            ws_clauses = segment_clauses(website_text, f"{pair_id}_website")
            vs_clauses = segment_clauses(vendor_text, f"{pair_id}_vendor")
            _dd.dump_json(stage_path, "stage_1_clauses.json", {
                "website": [
                    {"section": c.section_header, "text": c.text[:400]} for c in ws_clauses
                ],
                "vendor": [
                    {"section": c.section_header, "text": c.text[:400]} for c in vs_clauses
                ],
            })
        except Exception as exc:  # pragma: no cover
            print(f"  [debug-dump] stage_1_clauses skipped: {exc}")

    website_statements = extract_pps_from_policy(
        website_text,
        policy_source="first_party",
        policy_id=f"{pair_id}_website",
    )
    vendor_statements = extract_pps_from_policy(
        vendor_text,
        policy_source=f"third_party:{vendor_name}",
        policy_id=f"{pair_id}_vendor",
    )
    if stage_path is not None:
        _dd.dump_json(stage_path, "stage_1_raw_pps.json", {
            "website": _dd.summarize_statements(website_statements),
            "vendor": _dd.summarize_statements(vendor_statements),
        })

    # Validate vendor policy quality
    MIN_VENDOR_STATEMENTS = 3
    if len(vendor_statements) < MIN_VENDOR_STATEMENTS:
        print(
            f"\n  [WARNING] Vendor policy produced only {len(vendor_statements)} PPS "
            f"(minimum {MIN_VENDOR_STATEMENTS}). This may indicate a non-privacy "
            f"policy (e.g., AUP, ToS) or extraction failure."
        )
        if len(vendor_statements) == 0:
            print("  [SKIP] No vendor statements extracted — skipping this pair.")
            return None

    # Derive first-party names for actor normalization
    website_fp_names = _derive_first_party_names(pair_id, website_text)
    vendor_fp_names = [vendor_name] + [n.strip() for n in vendor_name.split("/") if n.strip()]

    _step_end("extract")

    _step_start("graph_build")
    print("\n[Step 2] Building knowledge graphs...")
    website_graph = build_graph(website_statements, "website", first_party_names=website_fp_names)
    vendor_graph = build_graph(vendor_statements, vendor_name, first_party_names=vendor_fp_names)

    _step_end("graph_build")

    _step_start("merge")
    print("\n[Step 3] Merging graphs...")
    merged_graph = merge_graphs(website_graph, vendor_graph, vendor_name, service_type)

    if stage_path is not None:
        def _graph_summary(g):
            node_types: dict[str, int] = {}
            edge_types: dict[str, int] = {}
            data_nodes: list[str] = []
            for nid, attrs in g.nodes(data=True):
                nt = attrs.get("node_type", "?")
                node_types[nt] = node_types.get(nt, 0) + 1
                if nt == "DataType":
                    data_nodes.append(attrs.get("label", nid))
            for _, _, a in g.edges(data=True):
                et = a.get("edge_type", "?")
                edge_types[et] = edge_types.get(et, 0) + 1
            return {
                "num_nodes": g.number_of_nodes(),
                "num_edges": g.number_of_edges(),
                "node_types": node_types,
                "edge_types": edge_types,
                "data_nodes": sorted(set(data_nodes)),
            }
        _dd.dump_json(stage_path, "stage_2_graph_stats.json", {
            "website": _graph_summary(website_graph),
            "vendor": _graph_summary(vendor_graph),
        })
        merged_summary = _graph_summary(merged_graph)
        aligned = []
        for u, v, a in merged_graph.edges(data=True):
            if a.get("edge_type") == "ALIGNED_TO":
                aligned.append({
                    "website_node": u,
                    "vendor_node": v,
                    "website_label": merged_graph.nodes[u].get("label", ""),
                    "vendor_label": merged_graph.nodes[v].get("label", ""),
                    "relation": a.get("relation"),
                })
        bridge_nodes = [n for n, a in merged_graph.nodes(data=True) if a.get("node_type") == "Bridge"]
        merged_summary["aligned_to_edges"] = aligned
        merged_summary["bridge_nodes"] = bridge_nodes
        _dd.dump_json(stage_path, "stage_3_merged_graph_stats.json", merged_summary)

        # Stage 4: neighborhoods (per data type).
        try:
            from graph_neighborhoods import (
                get_data_neighborhoods,
                get_aligned_pairs,
                get_bridge_flows,
                get_neighborhood_context,
            )
            w_nhs = get_data_neighborhoods(merged_graph, "1p", expand_ancestors=False)
            v_nhs = get_data_neighborhoods(merged_graph, "3p", expand_ancestors=False)
            ap = get_aligned_pairs(merged_graph)
            bridge_flows = get_bridge_flows(merged_graph, vendor_name)

            def _nh_summary(nh):
                return {
                    "data_type": nh.data_type,
                    "policy_source": nh.policy_source,
                    "n_statements": len(nh.statements),
                    "node_ids": nh.data_node_ids[:10],
                    "context": get_neighborhood_context(nh),
                    "statement_ids": [s.id for s in nh.statements],
                }

            _dd.dump_json(stage_path, "stage_4_neighborhoods.json", {
                "website_neighborhoods": [_nh_summary(n) for n in w_nhs],
                "vendor_neighborhoods": [_nh_summary(n) for n in v_nhs],
                "aligned_pairs": [
                    {
                        "data_type": pair.website.data_type,
                        "alignment_relation": pair.alignment_relation,
                        "website_n_statements": len(pair.website.statements),
                        "vendor_n_statements": len(pair.vendor.statements),
                        "website_statement_ids": [s.id for s in pair.website.statements],
                        "vendor_statement_ids": [s.id for s in pair.vendor.statements],
                    }
                    for pair in ap
                ],
                "bridge_flows": [
                    {
                        "data_type": bf.data_type,
                        "website_pps_id": bf.website_pps.id,
                        "vendor_pps_id": bf.vendor_pps.id,
                    }
                    for bf in bridge_flows
                ],
            })
        except Exception as exc:  # pragma: no cover
            print(f"  [debug-dump] stage_4 skipped: {exc}")

    _step_end("merge")

    _step_start("metrics")
    print("\n[Step 4] Computing graph metrics...")
    website_metrics = compute_graph_metrics(website_graph)
    vendor_metrics = compute_graph_metrics(vendor_graph)
    merged_metrics = compute_graph_metrics(merged_graph)
    print(
        f"  Website: {website_metrics['total_nodes']} nodes, {website_metrics['total_edges']} edges"
    )
    print(
        f"  Vendor:  {vendor_metrics['total_nodes']} nodes, {vendor_metrics['total_edges']} edges"
    )
    print(
        f"  Merged:  {merged_metrics['total_nodes']} nodes, {merged_metrics['total_edges']} edges"
    )

    _step_end("metrics")

    _step_start("patterns")
    print("\n[Step 5] Running inconsistency detection patterns...")
    all_inconsistencies = run_all_patterns(merged_graph, cross_policy=True)

    if stage_path is not None:
        try:
            trace = _dd.get_trace(pair_id)
            payload = {
                pattern: events
                for pattern, events in trace.items()
            }
            payload["_findings_emitted"] = [
                {
                    "pattern_id": inc.pattern_id,
                    "id": inc.inconsistency_id,
                    "verdict": inc.verdict.value if hasattr(inc.verdict, "value") else inc.verdict,
                    "severity": inc.severity.value if hasattr(inc.severity, "value") else inc.severity,
                    "s1_id": inc.statement_1.id,
                    "s2_id": inc.statement_2.id,
                    "s1_data": inc.statement_1.data_object,
                    "s2_data": inc.statement_2.data_object,
                    "s1_action": inc.statement_1.action,
                    "s2_action": inc.statement_2.action,
                    "explanation": inc.explanation[:300],
                }
                for inc in all_inconsistencies
            ]
            _dd.dump_json(stage_path, "stage_5_pattern_candidates.json", payload)
        except Exception as exc:  # pragma: no cover
            print(f"  [debug-dump] stage_5 skipped: {exc}")

    # Step 5b: mandatory LLM verification of every inconsistency candidate.
    # The verifier receives the full policy texts so it can read the
    # surrounding paragraph for each clause — mirroring the 5-agent deep
    # audit protocol. Every candidate is kept in the output with its verdict
    # set to one of {inconsistent, underspecified, non_conflict}; nothing
    # is dropped here. Downstream consumers (curator, CSV, audit) decide how
    # to use the verdict.
    _step_end("patterns")

    _step_start("verify")
    print(f"\n[Step 5b] Verifying {len(all_inconsistencies)} candidates with LLM...")
    candidates_as_dicts = [inc.to_dict() for inc in all_inconsistencies]
    for c in candidates_as_dicts:
        c["pair_id"] = pair_id
    policy_texts = {
        pair_id: {"first_party": website_text, "third_party": vendor_text}
    }
    verified_dicts = verify_candidates(
        candidates_as_dicts,
        max_workers=int(os.environ.get("VERIFIER_WORKERS", "8")),
        policy_texts=policy_texts,
    )
    if stage_path is not None:
        try:
            _dd.dump_json(stage_path, "stage_6_verifier.json", {
                "n_candidates": len(candidates_as_dicts),
                "raw_verdicts": [
                    {
                        "pattern_id": c.get("pattern_id"),
                        "s1_text": (c.get("statement_1", {}) or {}).get("source_text", "")[:200],
                        "s2_text": (c.get("statement_2", {}) or {}).get("source_text", "")[:200],
                        "llm_verdict": v.get("llm_verdict"),
                        "llm_false_alarm_category": v.get("llm_false_alarm_category"),
                        "llm_verified": v.get("llm_verified"),
                        "llm_confidence": v.get("llm_confidence"),
                        "llm_explanation": (v.get("llm_explanation") or "")[:300],
                    }
                    for c, v in zip(candidates_as_dicts, verified_dicts)
                ],
            })
        except Exception as exc:  # pragma: no cover
            print(f"  [debug-dump] stage_6 skipped: {exc}")
    # The verifier now returns the final 3-label verdict directly, so we
    # do NOT overwrite inc.verdict (the pattern-level severity-driver).
    # `inc.llm_verdict` is the authoritative post-verification label.
    verdict_counts = {"inconsistent": 0, "unspecified": 0, "non_conflict": 0}
    llm_verified_count = 0
    for inc, vd in zip(all_inconsistencies, verified_dicts):
        llm_verdict = str(vd.get("llm_verdict", "") or "")
        if llm_verdict in verdict_counts:
            verdict_counts[llm_verdict] += 1
        # Persist full LLM trace. inc.verdict (pattern-level Verdict enum)
        # remains unchanged — severity ranking during curation uses it.
        inc.llm_verified = bool(vd.get("llm_verified", False))
        inc.llm_confidence = str(vd.get("llm_confidence", "") or "")
        inc.llm_false_alarm_category = str(vd.get("llm_false_alarm_category", "none") or "none")
        inc.llm_verdict = llm_verdict
        inc.llm_explanation = str(vd.get("llm_explanation", "") or "")
        # Mirror the LLM explanation onto the public `explanation` only
        # when the verifier agrees the finding stands; for non_conflict
        # we keep the pattern's explanation so the CSV preserves what
        # the structural detector originally reported.
        if llm_verdict in ("inconsistent", "unspecified"):
            inc.explanation = inc.llm_explanation or inc.explanation
        if inc.llm_verified:
            llm_verified_count += 1
    print(
        f"  Verified: {len(verified_dicts)} candidates "
        f"(inconsistent={verdict_counts['inconsistent']}, "
        f"unspecified={verdict_counts['unspecified']}, "
        f"non_conflict={verdict_counts['non_conflict']}, "
        f"llm_verified={llm_verified_count}/{len(verified_dicts)})"
    )

    website_inconsistencies = [
        inconsistency
        for inconsistency in all_inconsistencies
        if inconsistency.statement_1.policy_source == "first_party"
        and inconsistency.statement_2.policy_source == "first_party"
    ]
    vendor_policy_source = f"third_party:{vendor_name}"
    vendor_inconsistencies = [
        inconsistency
        for inconsistency in all_inconsistencies
        if inconsistency.statement_1.policy_source == vendor_policy_source
        and inconsistency.statement_2.policy_source == vendor_policy_source
    ]

    _step_end("verify")

    _step_start("visualize")
    print("\n[Step 6] Generating visualizations...")
    visualize_graph(
        website_graph,
        output_path / f"{pair_id}_website_graph.html",
        title=f"{pair_id} - Website Policy",
        inconsistencies=website_inconsistencies,
    )
    visualize_graph(
        vendor_graph,
        output_path / f"{pair_id}_vendor_graph.html",
        title=f"{pair_id} - {vendor_name} Policy",
        inconsistencies=vendor_inconsistencies,
    )
    visualize_graph(
        merged_graph,
        output_path / f"{pair_id}_merged_graph.html",
        title=f"{pair_id} - Merged ({vendor_name})",
        inconsistencies=all_inconsistencies,
    )

    _step_end("visualize")

    # ── GDPR completeness analysis (per-policy, post-extraction)
    # Each PPS carries a single-label gdpr_categories after the 2026-04-18
    # argmax switch, so this aggregation gives a clean 1-of-18 coverage
    # signal per policy. Computed here — no LLM calls.
    #
    # We track TWO coverage signals per side:
    #   * pps_coverage (flat fields `covered/missing/coverage_pct/...`) —
    #     bounded by extraction output; reflects what the pipeline surfaced.
    #   * clause_coverage — runs RoBERTa on every segmented clause
    #     regardless of whether the LLM emitted a PPS for it, giving the
    #     policy-level disclosure upper bound.
    #   * extraction_gap — delta between the two; categories present in the
    #     policy text that never reached a PPS (usually rights language,
    #     DPO/controller contacts, lodge-complaint, adequacy etc.).
    #
    # Computed BEFORE the save step because the pattern→Article 13/14
    # attribution below consumes `website_pps_cov`/`vendor_pps_cov` (it
    # needs each side's `missing` list to flag coverage gaps), and
    # curated_inconsistencies must see the attribution on every finding.
    website_pps_cov    = compute_gdpr_completeness(website_statements)
    vendor_pps_cov     = compute_gdpr_completeness(vendor_statements)
    website_clause_cov = compute_clause_gdpr_coverage(website_text, f"{pair_id}_website")
    vendor_clause_cov  = compute_clause_gdpr_coverage(vendor_text,  f"{pair_id}_vendor")
    website_pps_cov["clause_coverage"]  = website_clause_cov
    website_pps_cov["extraction_gap"]   = compute_extraction_gap(website_pps_cov, website_clause_cov)
    vendor_pps_cov["clause_coverage"]   = vendor_clause_cov
    vendor_pps_cov["extraction_gap"]    = compute_extraction_gap(vendor_pps_cov,  vendor_clause_cov)
    gdpr_completeness = {
        "website": website_pps_cov,
        "vendor":  vendor_pps_cov,
    }
    # Cross-policy diff — surfaces asymmetries the per-side summaries can't,
    # e.g. a website that promises DSR categories the vendor never discloses.
    # Pure aggregation; no LLM calls. Uses PPS-level covered/missing (the
    # flat keys on each side dict), so the clause_coverage/extraction_gap
    # sub-keys don't interfere with the existing behaviour.
    gdpr_comparison = compare_gdpr_completeness(
        gdpr_completeness["website"],
        gdpr_completeness["vendor"],
    )

    # Pattern → Article 13/14 attribution: for every cross-policy finding,
    # record the categories the pattern structurally breaches plus whether
    # the primary category is also a completeness gap on one side. Serves
    # as the bridge between the inconsistency output and the GDPR
    # completeness output; downstream CSVs and paper tables can then slice
    # findings by GDPR category. Pure aggregation; no LLM calls.
    for inconsistency in all_inconsistencies:
        inconsistency.gdpr_attribution = attribute_finding_to_gdpr(
            pattern_id=inconsistency.pattern_id,
            finding_gdpr_categories=inconsistency.gdpr_categories,
            website_completeness=website_pps_cov,
            vendor_completeness=vendor_pps_cov,
        )

    _step_start("save")
    print("\n[Step 7] Saving results...")
    summary_counts = _summary_counts(all_inconsistencies)
    curated_inconsistencies = _curate_inconsistencies(
        [inconsistency.to_dict() for inconsistency in all_inconsistencies]
    )
    curated_summary = _summary_counts(curated_inconsistencies)
    repro_meta = {
        # Extraction backend
        "extraction_backend": EXTRACTION_BACKEND,
        "extraction_prompt_version": EXTRACTION_PROMPT_VERSION,
        "scope_prompt_version": SCOPE_PROMPT_VERSION,
        "verifier_prompt_version": VERIFIER_PROMPT_VERSION,
        "cluster_verify_prompt_version": CLUSTER_VERIFY_PROMPT_VERSION,
        # Gates the extractor's _build_pps post-processing cascade. Included so
        # reviewers can tell pre- and post-v12 pipeline outputs apart without
        # inspecting code. See audit F27.
        "build_pps_version": BUILD_PPS_VERSION,
        "reflection_enabled": EXTRACTION_REFLECTION_ENABLED,
        "reflection_rounds": EXTRACTION_REFLECTION_ROUNDS,
        "temperature": EXTRACTION_TEMPERATURE,
        # Code + config identity so results can be reproduced
        "commit_sha": _COMMIT_SHA,
        "normalizer_ontology_hash": _ONTOLOGY_HASH,
        "patterns_version": _PATTERNS_VERSION,
    }
    if EXTRACTION_BACKEND == "llamacpp":
        repro_meta["llamacpp_base_url"] = LLAMACPP_BASE_URL
        repro_meta["llamacpp_model_name"] = LLAMACPP_MODEL_NAME

    results = {
        "pair_id": pair_id,
        "vendor_name": vendor_name,
        "service_type": service_type,
        "timestamp": datetime.now().isoformat(),
        # `_pair_total` is assigned *after* the result dict is built (see
        # the final print("--- Timing ---") block), so we can't reference
        # it here — compute the elapsed seconds inline against _pair_t0
        # (the monotonic clock captured at run_pair entry). step_timings
        # reads already-populated entries ("_start" keys are the open
        # markers, "<name>" keys are the closed durations).
        "elapsed_s": round(_time.monotonic() - _pair_t0, 2),
        "step_timings": {k: round(v, 2) for k, v in _step_timings.items()
                         if not k.endswith("_start")},
        "reproducibility": repro_meta,
        "metrics": {
            "website": website_metrics,
            "vendor": vendor_metrics,
            "merged": merged_metrics,
        },
        "gdpr_completeness": gdpr_completeness,
        "gdpr_comparison": gdpr_comparison,
        "website_statements": [statement.to_dict() for statement in website_statements],
        "vendor_statements": [statement.to_dict() for statement in vendor_statements],
        "inconsistencies": [inconsistency.to_dict() for inconsistency in all_inconsistencies],
        "curated_inconsistencies": curated_inconsistencies,
        "summary": {
            "total_website_statements": len(website_statements),
            "total_vendor_statements": len(vendor_statements),
            "total_inconsistencies": len(all_inconsistencies),
            **summary_counts,
        },
        "curated_summary": {
            "total_inconsistencies": len(curated_inconsistencies),
            **curated_summary,
        },
    }

    json_path = output_path / f"{pair_id}_results.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"  Results JSON: {json_path}")

    # Also emit a standalone GDPR-completeness artifact per pair so
    # downstream aggregation scripts / reviewers can load completeness
    # without parsing the full results JSON.
    gdpr_path = output_path / f"{pair_id}_gdpr_completeness.json"
    with gdpr_path.open("w", encoding="utf-8") as handle:
        json.dump({
            "pair_id": pair_id,
            "vendor_name": vendor_name,
            "timestamp": datetime.now().isoformat(),
            "gdpr_completeness": gdpr_completeness,
            "gdpr_comparison": gdpr_comparison,
        }, handle, indent=2)
    print(f"  GDPR completeness JSON: {gdpr_path}")

    # Seed the pair-level cache so future runs hit even if this output dir
    # is archived.
    try:
        if save_pair_cache(_pcache_key, results):
            results.setdefault("pair_cache", {})
            results["pair_cache"].update({"hit": False, "key": _pcache_key, "seeded": True})
    except Exception as exc:
        print(f"  [pair-cache] save skipped: {exc}")

    _step_end("save")
    _pair_total = _time.monotonic() - _pair_t0

    print(f"\n{'=' * 60}")
    print(f"SUMMARY - {pair_id}")
    print(f"{'=' * 60}")
    print(f"  Website statements: {len(website_statements)}")
    print(f"  Vendor statements:  {len(vendor_statements)}")
    print(f"  Inconsistencies:    {len(all_inconsistencies)}")
    print(f"  Curated candidates: {len(curated_inconsistencies)}")
    for side in ("website", "vendor"):
        gc = gdpr_completeness[side]
        flag = "COMPLETE" if gc["is_complete"] else f"INCOMPLETE (missing {len(gc['missing'])}/18)"
        print(f"  GDPR {side:<7}:    {gc['coverage_pct']:.1f}% coverage — {flag}")
        if gc["missing"]:
            # Keep the log compact — just the category names, comma-separated.
            print(f"    missing: {', '.join(gc['missing'])}")

    # Cross-policy GDPR diff — flag the 4 asymmetry patterns only when present.
    _flags = gdpr_comparison["asymmetry_flags"]
    _active_flags = [name for name, active in _flags.items() if active]
    print(
        f"  GDPR diff:      Δ={gdpr_comparison['coverage_pct_delta']:+.1f}pp "
        f"(w>v: {len(gdpr_comparison['only_website_covers'])}, "
        f"v>w: {len(gdpr_comparison['only_vendor_covers'])}, "
        f"shared miss: {len(gdpr_comparison['both_miss'])}, "
        f"flags: {len(_active_flags)}/{len(_flags)})"
    )
    if _active_flags:
        print(f"    asymmetries: {', '.join(_active_flags)}")

    for severity in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        count = summary_counts["by_severity"][severity]
        if count:
            print(f"    {severity}: {count}")
    print(f"  --- Timing ---")
    for step_name in ["extract", "graph_build", "merge", "metrics", "patterns", "verify", "visualize", "save"]:
        if step_name in _step_timings:
            print(f"    {step_name:<14s}: {_step_timings[step_name]:>7.1f}s")
    print(f"    {'TOTAL':<14s}: {_pair_total:>7.1f}s")
    print(f"{'=' * 60}")

    return results


def _write_ground_truth_csv(all_results: list[dict], output_path: Path, curated: bool) -> int:
    fieldnames = [
        "pair_id",
        "inconsistency_id",
        "pattern_id",
        "pattern_name",
        "statement_1_id",
        "statement_1_text",
        "statement_2_id",
        "statement_2_text",
        "auto_verdict",
        "auto_severity",
        "gdpr_categories",
        "explanation",
        "MANUAL_VERDICT",
        "MANUAL_SEVERITY",
        "MANUAL_NOTES",
    ]

    total = 0
    key = "curated_inconsistencies" if curated else "inconsistencies"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            for inconsistency in result[key]:
                total += 1
                writer.writerow(
                    {
                        "pair_id": result["pair_id"],
                        "inconsistency_id": inconsistency["inconsistency_id"],
                        "pattern_id": inconsistency["pattern_id"],
                        "pattern_name": inconsistency["pattern_name"],
                        "statement_1_id": inconsistency["statement_1"]["id"],
                        "statement_1_text": inconsistency["statement_1"].get("source_text", "")[:500],
                        "statement_2_id": inconsistency["statement_2"]["id"],
                        "statement_2_text": inconsistency["statement_2"].get("source_text", "")[:500],
                        "auto_verdict": inconsistency["verdict"],
                        "auto_severity": inconsistency["severity"],
                        "gdpr_categories": "; ".join(inconsistency.get("gdpr_categories", [])),
                        "explanation": inconsistency["explanation"],
                        "MANUAL_VERDICT": "",
                        "MANUAL_SEVERITY": "",
                        "MANUAL_NOTES": "",
                    }
                )
    return total


def _extract_sentence_fields(inconsistency: dict) -> tuple[str, str]:
    first = inconsistency["statement_1"]
    second = inconsistency["statement_2"]
    first_source = first.get("policy_source", "")
    second_source = second.get("policy_source", "")

    website_sentence = ""
    third_party_sentence = ""

    if first_source == "first_party":
        website_sentence = first.get("source_text", "")
    elif first_source.startswith("third_party:"):
        third_party_sentence = first.get("source_text", "")

    if second_source == "first_party" and not website_sentence:
        website_sentence = second.get("source_text", "")
    elif second_source.startswith("third_party:") and not third_party_sentence:
        third_party_sentence = second.get("source_text", "")

    # Fallback for intra-policy rows (both statements from the same policy):
    # the policy_source logic above only populates one field, leaving the other blank.
    # Populate both from statement_1 and statement_2 directly so annotation rows are never empty.
    if not website_sentence:
        website_sentence = first.get("source_text", "")
    if not third_party_sentence:
        third_party_sentence = second.get("source_text", "")

    return website_sentence[:500], third_party_sentence[:500]


def _write_sentence_annotation_csv(
    all_results: list[dict],
    output_path: Path,
    curated: bool,
    cross_policy_only: bool = False,
) -> int:
    fieldnames = [
        "pair_id",
        "inconsistency_id",
        "pattern_id",
        "severity",
        "website_sentence",
        "third_party_sentence",
        "gdpr_category",
        "label",
        "justification",
    ]

    total = 0
    key = "curated_inconsistencies" if curated else "inconsistencies"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            for inconsistency in result[key]:
                website_sentence, third_party_sentence = _extract_sentence_fields(inconsistency)
                if cross_policy_only and (not website_sentence or not third_party_sentence):
                    continue
                categories = inconsistency.get("gdpr_categories", []) or [""]
                for category in categories:
                    total += 1
                    writer.writerow(
                        {
                            "pair_id": result["pair_id"],
                            "inconsistency_id": inconsistency["inconsistency_id"],
                            "pattern_id": inconsistency["pattern_id"],
                            "severity": inconsistency["severity"],
                            "website_sentence": website_sentence,
                            "third_party_sentence": third_party_sentence,
                            "gdpr_category": category,
                            "label": "inconsistency",
                            "justification": inconsistency["explanation"],
                        }
                    )
    return total


def _generate_ground_truth_csv(all_results: list[dict], output_dir: Path) -> None:
    curated_path = output_dir / "ground_truth_template.csv"
    raw_path = output_dir / "ground_truth_template_raw.csv"
    sentence_curated_path = output_dir / "sentence_annotation_template.csv"
    sentence_raw_path = output_dir / "sentence_annotation_template_raw.csv"
    cross_sentence_curated_path = output_dir / "cross_policy_sentence_annotation_template.csv"
    cross_sentence_raw_path = output_dir / "cross_policy_sentence_annotation_template_raw.csv"
    curated_total = _write_ground_truth_csv(all_results, curated_path, curated=True)
    raw_total = _write_ground_truth_csv(all_results, raw_path, curated=False)
    sentence_curated_total = _write_sentence_annotation_csv(
        all_results,
        sentence_curated_path,
        curated=True,
    )
    sentence_raw_total = _write_sentence_annotation_csv(
        all_results,
        sentence_raw_path,
        curated=False,
    )
    cross_sentence_curated_total = _write_sentence_annotation_csv(
        all_results,
        cross_sentence_curated_path,
        curated=True,
        cross_policy_only=True,
    )
    cross_sentence_raw_total = _write_sentence_annotation_csv(
        all_results,
        cross_sentence_raw_path,
        curated=False,
        cross_policy_only=True,
    )
    print(f"\n  Ground truth template: {curated_path} ({curated_total} rows)")
    print(f"  Raw ground truth template: {raw_path} ({raw_total} rows)")
    print(f"  Sentence annotation template: {sentence_curated_path} ({sentence_curated_total} rows)")
    print(f"  Raw sentence annotation template: {sentence_raw_path} ({sentence_raw_total} rows)")
    print(
        f"  Cross-policy sentence annotation template: {cross_sentence_curated_path} ({cross_sentence_curated_total} rows)"
    )
    print(
        f"  Raw cross-policy sentence annotation template: {cross_sentence_raw_path} ({cross_sentence_raw_total} rows)"
    )


def _generate_research_candidate_csv(
    all_results: list[dict],
    output_dir: Path,
    max_total: int = 100,
) -> int:
    fieldnames = [
        "rank",
        "pair_id",
        "vendor_name",
        "pattern_id",
        "pattern_name",
        "severity",
        "verdict",
        "llm_verdict",
        "llm_confidence",
        "website_sentence",
        "third_party_sentence",
        "statement_1_text",
        "statement_2_text",
        "gdpr_categories",
        "justification",
    ]

    flattened: list[dict] = []
    for result in all_results:
        for inconsistency in result.get("inconsistencies", []):
            flattened.append(
                {
                    **inconsistency,
                    "pair_id": result["pair_id"],
                    "vendor_name": result["vendor_name"],
                }
            )

    ordered = sorted(
        flattened,
        key=lambda item: (
            -_inconsistency_priority(item)[0],
            -_inconsistency_priority(item)[1],
            -_inconsistency_priority(item)[2],
            -_inconsistency_priority(item)[3],
            item["pair_id"],
            item["inconsistency_id"],
        ),
    )

    per_pair_cap = 35
    # Round 9/11: Π₃/Π₆/Π₇/Π₉/Π₁₀ removed. Caps shared with _curate_inconsistencies
    # via PATTERN_FINDING_CAPS (unified 2026-04-18 — previously drifted).
    per_pattern_caps = PATTERN_FINDING_CAPS
    # Relaxed-fill cap: 1.5x to backfill if primary patterns run short on unique candidates.
    relaxed_pattern_caps = {k: int(v * 1.5) for k, v in per_pattern_caps.items()}
    pair_counts: dict[str, int] = {}
    pattern_counts: dict[str, int] = {}
    selected: list[dict] = []
    seen_sentence_pairs: set[tuple[str, str, str]] = set()

    def passes_quality_gates(item: dict, relaxed_caps: bool = False) -> bool:
        pair_id = item["pair_id"]
        pattern_id = item["pattern_id"]
        same_policy = (
            item["statement_1"].get("policy_source", "")
            == item["statement_2"].get("policy_source", "")
        )
        same_source_clause = (
            item["statement_1"].get("source_text", "").strip()
            and item["statement_1"].get("source_text", "").strip()
            == item["statement_2"].get("source_text", "").strip()
        )
        if same_policy and same_source_clause and pattern_id != "Π₁":
            return False
        if not relaxed_caps and pair_counts.get(pair_id, 0) >= per_pair_cap:
            return False
        effective_cap = (
            per_pattern_caps.get(pattern_id, 20)
            if not relaxed_caps
            else relaxed_pattern_caps.get(pattern_id, 30)
        )
        if pattern_counts.get(pattern_id, 0) >= effective_cap:
            return False
        sentence_key = (
            pattern_id,
            item["statement_1"].get("source_text", "").strip(),
            item["statement_2"].get("source_text", "").strip(),
            item.get("explanation", "").strip(),
        )
        if sentence_key in seen_sentence_pairs:
            return False
        # Π₁: filter out LLM hallucinations where the extracted data_object
        # does not actually appear in the prohibition's source text.
        # This catches cases where the extractor assigned a data type from a nearby clause.
        if pattern_id == "Π₁":
            s1 = item["statement_1"]
            data_obj = s1.get("data_object", "").lower()
            src = s1.get("source_text", "").lower()
            # Use the most distinctive word(s) from the data object (≥4 chars) as proxies.
            distinctive_words = sorted(
                [w for w in data_obj.split() if len(w) >= 4], key=len, reverse=True
            )
            if distinctive_words and not any(w in src for w in distinctive_words[:2]):
                return False

        return True

    def register_selection(item: dict) -> None:
        pair_id = item["pair_id"]
        pattern_id = item["pattern_id"]
        sentence_key = (
            pattern_id,
            item["statement_1"].get("source_text", "").strip(),
            item["statement_2"].get("source_text", "").strip(),
            item.get("explanation", "").strip(),
        )
        seen_sentence_pairs.add(sentence_key)
        pair_counts[pair_id] = pair_counts.get(pair_id, 0) + 1
        pattern_counts[pattern_id] = pattern_counts.get(pattern_id, 0) + 1

    for item in ordered:
        if not passes_quality_gates(item, relaxed_caps=False):
            continue

        selected.append(item)
        register_selection(item)
        if len(selected) >= max_total:
            break

    # Select extra headroom so that after LLM verification drops non_conflict rows
    # we still have enough to fill max_total.  3x gives buffer for the ~30-40% rejection rate.
    selection_target = int(max_total * 3)
    if len(selected) < selection_target:
        for item in ordered:
            if item in selected:
                continue
            if not passes_quality_gates(item, relaxed_caps=True):
                continue
            selected.append(item)
            register_selection(item)
            if len(selected) >= selection_target:
                break

    # No re-verification here. Every candidate already carries
    # `llm_verdict` / `llm_explanation` / `llm_confidence` from the
    # per-pair verifier (run_pair → [Step 5b] → verify_candidates).
    # Re-running verify_candidates(selected) here would re-pay ~300 LLM
    # calls per shard with no new information.
    final: list[dict] = list(selected)[:max_total]

    output_path = output_dir / "research_grade_candidates_top100.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, item in enumerate(final, start=1):
            website_sentence, third_party_sentence = _extract_sentence_fields(item)
            # Use LLM verdict and explanation when available; fall back to pattern's.
            display_verdict = item.get("llm_verdict") or item.get("verdict", "")
            display_explanation = item.get("llm_explanation") or item.get("explanation", "")
            writer.writerow(
                {
                    "rank": index,
                    "pair_id": item["pair_id"],
                    "vendor_name": item["vendor_name"],
                    "pattern_id": item["pattern_id"],
                    "pattern_name": item["pattern_name"],
                    "severity": item["severity"],
                    "verdict": display_verdict,
                    "llm_verdict": item.get("llm_verdict", ""),
                    "llm_confidence": item.get("llm_confidence", ""),
                    "website_sentence": website_sentence,
                    "third_party_sentence": third_party_sentence,
                    "statement_1_text": item["statement_1"].get("source_text", "")[:500],
                    "statement_2_text": item["statement_2"].get("source_text", "")[:500],
                    "gdpr_categories": "; ".join(item.get("gdpr_categories", [])),
                    "justification": display_explanation[:600],
                }
            )

    print(f"  Research-grade candidate CSV: {output_path} ({len(final)} rows)")
    return len(final)


def _generate_all_findings_csv(all_results: list[dict],
                               output_dir: Path) -> int:
    """Write **every** finding from every pair to a single CSV, with no
    per-pair / per-pattern caps and no top-N truncation. Intended as the
    primary production artefact for the full top-10 / 1.2k-pair runs.

    Columns: pattern identity, severity (pattern-level), the LLM
    verifier's 3-label verdict (``llm_verdict`` ∈ {inconsistent,
    unspecified, non_conflict}), and the source clauses. Findings are
    emitted in the same priority order (``-_inconsistency_priority``)
    used by the curator, so the first row is the most severe / most
    important.
    """
    fieldnames = [
        "pair_id", "vendor_name", "pattern_id", "pattern_name",
        "severity", "verdict",
        "llm_verdict", "llm_confidence", "llm_explanation",
        "llm_verified", "llm_false_alarm_category",
        "website_sentence", "third_party_sentence",
        "statement_1_text", "statement_2_text",
        "gdpr_categories", "explanation",
    ]

    flattened: list[dict] = []
    for result in all_results:
        for inc in result.get("inconsistencies", []):
            flattened.append({
                **inc,
                "pair_id": result["pair_id"],
                "vendor_name": result["vendor_name"],
            })

    ordered = sorted(
        flattened,
        key=lambda item: (
            -_inconsistency_priority(item)[0],
            -_inconsistency_priority(item)[1],
            -_inconsistency_priority(item)[2],
            -_inconsistency_priority(item)[3],
            item["pair_id"],
            item.get("inconsistency_id", ""),
        ),
    )

    out = output_dir / "all_findings.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in ordered:
            ws, tps = _extract_sentence_fields(item)
            writer.writerow({
                "pair_id":              item["pair_id"],
                "vendor_name":          item["vendor_name"],
                "pattern_id":           item.get("pattern_id", ""),
                "pattern_name":         item.get("pattern_name", ""),
                "severity":             item.get("severity", ""),
                "verdict":              item.get("verdict", ""),
                "llm_verdict":          item.get("llm_verdict", ""),
                "llm_confidence":       item.get("llm_confidence", ""),
                "llm_explanation":      (item.get("llm_explanation") or "")[:600],
                "llm_verified":         item.get("llm_verified", ""),
                "llm_false_alarm_category": item.get("llm_false_alarm_category", ""),
                "website_sentence":     ws,
                "third_party_sentence": tps,
                "statement_1_text":     (item.get("statement_1") or {}).get("source_text", "")[:500],
                "statement_2_text":     (item.get("statement_2") or {}).get("source_text", "")[:500],
                "gdpr_categories":      "; ".join(item.get("gdpr_categories", [])),
                "explanation":          (item.get("explanation") or "")[:600],
            })

    print(f"  All-findings CSV: {out} ({len(ordered)} rows, no caps)")
    return len(ordered)


def _generate_aggregate_summary(all_results: list[dict], output_dir: Path) -> None:
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_pairs": len(all_results),
        "total_inconsistencies": sum(len(result["inconsistencies"]) for result in all_results),
        "total_curated_inconsistencies": sum(
            len(result.get("curated_inconsistencies", [])) for result in all_results
        ),
        "by_severity": {},
        "by_pattern": {},
        "by_verdict": {},
        "curated_by_severity": {},
        "curated_by_pattern": {},
        "curated_by_verdict": {},
        "per_pair": [],
    }

    for result in all_results:
        for severity, count in result["summary"]["by_severity"].items():
            summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + count
        for pattern_id, count in result["summary"]["by_pattern"].items():
            summary["by_pattern"][pattern_id] = summary["by_pattern"].get(pattern_id, 0) + count
        for verdict, count in result["summary"]["by_verdict"].items():
            summary["by_verdict"][verdict] = summary["by_verdict"].get(verdict, 0) + count
        for severity, count in result.get("curated_summary", {}).get("by_severity", {}).items():
            summary["curated_by_severity"][severity] = (
                summary["curated_by_severity"].get(severity, 0) + count
            )
        for pattern_id, count in result.get("curated_summary", {}).get("by_pattern", {}).items():
            summary["curated_by_pattern"][pattern_id] = (
                summary["curated_by_pattern"].get(pattern_id, 0) + count
            )
        for verdict, count in result.get("curated_summary", {}).get("by_verdict", {}).items():
            summary["curated_by_verdict"][verdict] = (
                summary["curated_by_verdict"].get(verdict, 0) + count
            )
        summary["per_pair"].append(
            {
                "pair_id": result["pair_id"],
                "vendor_name": result["vendor_name"],
                "website_stmts": result["summary"]["total_website_statements"],
                "vendor_stmts": result["summary"]["total_vendor_statements"],
                "inconsistencies": result["summary"]["total_inconsistencies"],
                "curated_inconsistencies": result.get("curated_summary", {}).get(
                    "total_inconsistencies", 0
                ),
            }
        )

    summary_path = output_dir / "aggregate_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"  Aggregate summary: {summary_path}")


def run_single_policy_gdpr(
    policy_text: str,
    policy_id: str = "policy",
    policy_source: str = "first_party",
    output_dir: str | Path | None = None,
) -> dict:
    """Run GDPR-completeness analysis on ONE privacy policy (no pair, no
    patterns, no verifier). Segment → extract → aggregate.

    Reports two coverage signals so an extractor miss cannot silently under-
    report a rights/legal disclosure that is present in the text:

      * ``gdpr_completeness`` (PPS-level) — categories formalised by the
        LLM extractor. Bounded by extraction quality.
      * ``gdpr_completeness.clause_coverage`` (segment-level) — categories
        emitted by the RoBERTa classifier over every segmented clause,
        regardless of whether the LLM surfaced a PPS for it. This is the
        policy-level disclosure upper bound.
      * ``gdpr_completeness.extraction_gap`` — delta between the two,
        highlighting categories present in the text but not surfaced as PPS
        (typically rights language, DPO contacts, lodge-complaint,
        adequacy).

    Cheaper than ``run_pair`` because it skips graph construction, pattern
    matching and LLM verification. Segment-level labels are cached after
    the first run (``gdpr_seg_<digest>.json``) so repeat calls stay
    sub-second. Useful when the research question is "does this policy
    disclose all 18 GDPR Art. 13/14 categories?" on a standalone policy.

    Outputs: writes ``<policy_id>_gdpr_completeness.json`` into
    ``output_dir`` and returns the same payload as a dict.
    """
    import time as _time
    t0 = _time.monotonic()

    output_path = Path(output_dir) if output_dir else OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"  GDPR completeness — single policy: {policy_id}")
    print("=" * 70)
    print(f"  policy_source: {policy_source}")
    print(f"  text length:   {len(policy_text)} chars")

    statements = extract_pps_from_policy(
        policy_text,
        policy_source=policy_source,
        policy_id=policy_id,
    )

    completeness = compute_gdpr_completeness(statements)
    # Segment-level upper bound: every clause is labelled by RoBERTa
    # regardless of whether the LLM emitted a PPS for it, so rights /
    # legal text that never reached a PPS still counts toward disclosure.
    clause_cov = compute_clause_gdpr_coverage(policy_text, policy_id)
    completeness["clause_coverage"] = clause_cov
    completeness["extraction_gap"] = compute_extraction_gap(completeness, clause_cov)

    elapsed = _time.monotonic() - t0
    result = {
        "policy_id": policy_id,
        "policy_source": policy_source,
        "timestamp": datetime.now().isoformat(),
        "reproducibility": {
            "commit_sha": _COMMIT_SHA,
            "ontology_hash": _ONTOLOGY_HASH,
            "patterns_version": _PATTERNS_VERSION,
            "extraction_prompt_version": EXTRACTION_PROMPT_VERSION,
            "scope_prompt_version": SCOPE_PROMPT_VERSION,
        },
        "n_statements": len(statements),
        "gdpr_completeness": completeness,
        "elapsed_s": round(elapsed, 2),
    }

    out_path = output_path / f"{policy_id}_gdpr_completeness.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  GDPR completeness JSON: {out_path}")

    flag = "COMPLETE" if completeness["is_complete"] else (
        f"INCOMPLETE (missing {len(completeness['missing'])}/18)"
    )
    print(f"  PPS coverage:     {completeness['coverage_pct']:.1f}% — {flag}")
    clause_pct = float(clause_cov.get("coverage_pct", 0.0))
    print(f"  Segment coverage: {clause_pct:.1f}% (disclosure upper bound)")
    gap_cats = completeness["extraction_gap"].get("extraction_gap_categories", [])
    if gap_cats:
        print(f"  Extraction gap:   {len(gap_cats)} categories "
              f"(disclosed in text, not formalised as PPS): {', '.join(gap_cats)}")
    if completeness["missing"]:
        print(f"  Missing (PPS):    {', '.join(completeness['missing'])}")
    for family, stats in completeness["per_family_coverage"].items():
        print(f"    {family:<25s} {stats['coverage_pct']:>5.1f}% "
              f"({len(stats['covered'])}/{stats['size']})")
    print(f"  Elapsed: {elapsed:.1f}s")
    return result


def run_batch(
    manifest_path: str | Path,
    output_dir: str | Path | None = None,
) -> list[dict]:
    """Process a manifest CSV of website/vendor pairs."""

    manifest = Path(manifest_path)
    output_path = Path(output_dir) if output_dir else OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    with manifest.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    print(f"\nBatch processing {len(rows)} policy pairs...")
    results: list[dict] = []

    for index, row in enumerate(rows, start=1):
        pair_id = row.get("pair_id") or f"pair_{index}"
        website_file = Path(row["website_file"])
        vendor_file = Path(row["vendor_file"])
        vendor_name = row["vendor_name"]
        service_type = row.get("service_type", "analytics")

        if not website_file.is_absolute():
            website_file = manifest.parent / website_file
        if not vendor_file.is_absolute():
            vendor_file = manifest.parent / vendor_file

        # Resume guard: if a previous shard already wrote the per-pair JSON,
        # reuse it instead of re-extracting. Lets us cancel + resubmit a
        # running shard to pick up code changes mid-corpus without losing
        # the work already on disk. The cached payload is loaded so the
        # aggregation passes still see every pair.
        existing_path = output_path / f"{pair_id}_results.json"
        if existing_path.exists():
            try:
                with existing_path.open("r", encoding="utf-8") as fh:
                    results.append(json.load(fh))
                print(f"\n  RESUME {pair_id}: loaded existing per-pair JSON")
                continue
            except Exception as exc:
                print(f"\n  WARN {pair_id}: existing JSON unreadable ({exc}); re-running")

        try:
            result = run_pair(
                website_file.read_text(encoding="utf-8"),
                vendor_file.read_text(encoding="utf-8"),
                vendor_name=vendor_name,
                service_type=service_type,
                output_dir=output_path,
                pair_id=pair_id,
            )
            if result is not None:
                results.append(result)
            else:
                print(f"\n  SKIPPED {pair_id}: insufficient vendor data")
        except Exception as exc:
            print(f"\n  ERROR processing {pair_id}: {exc}")

    # Production-mode aggregation order (Stage 2 was removed — the
    # per-pair verifier's 3-label llm_verdict is the authoritative
    # post-verification signal):
    # 1) Write the uncapped all-findings CSV — primary production output
    #    for the top10 / 1.2k-pair runs (documents every finding).
    # 2) Aggregate-summary, then the legacy curation artefacts (ground-
    #    truth annotation templates and the top-100 paper CSV).
    _generate_all_findings_csv(results, output_path)
    _generate_aggregate_summary(results, output_path)
    _generate_ground_truth_csv(results, output_path)
    _generate_research_candidate_csv(results, output_path)
    return results


def run_gdpr_pair_comparison(
    website_text: str,
    vendor_text: str,
    pair_id: str = "pair",
    website_id: str = "website",
    vendor_id: str = "vendor",
    output_dir: str | Path | None = None,
    second_role: str = "third_party",
) -> dict:
    """GDPR-completeness diff on two policies (no graph, no patterns).

    Extracts PPS from each policy, computes per-side completeness (both
    PPS-level and segment-level, with the extraction gap), and diffs the
    two sides via ``compare_gdpr_completeness`` — surfacing asymmetries
    like ``dsr_gap`` (website promises DSR categories the vendor never
    discloses) that the per-side summaries cannot produce alone.

    Segment-level coverage is attached to each side so a rights/legal
    disclosure present in the text but missed by PPS extraction does not
    silently under-report. Segment labels are cached after the first run,
    so adding this is free on repeat calls.

    Extraction roles
    ----------------
    ``second_role`` controls the actor context used when extracting PPS
    from the second policy. Defaults to ``"third_party"`` to preserve the
    website-vs-vendor behaviour, but can be set to ``"first_party"`` or
    ``"second_party"`` when the tool is used to compare two arbitrary
    policies rather than a true website/vendor pair. Accepted values are
    aligned with ``extract_pps_from_policy``'s ``policy_source`` tag:
    ``third_party`` / ``first_party`` / ``second_party``.

    Writes ``<pair_id>_gdpr_comparison.json`` into output_dir and returns
    the same payload.
    """
    import time as _time
    t0 = _time.monotonic()

    output_path = Path(output_dir) if output_dir else OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    allowed_roles = {"third_party", "first_party", "second_party"}
    if second_role not in allowed_roles:
        raise ValueError(
            f"second_role must be one of {sorted(allowed_roles)}; got {second_role!r}"
        )
    if second_role == "third_party":
        second_policy_source = f"third_party:{vendor_id}"
    else:
        # Plain role tag without the ``:name`` suffix, matching the rest
        # of the codebase's conventions for first-/second-party sides.
        second_policy_source = second_role

    print("\n" + "=" * 70)
    print(f"  GDPR-completeness diff — pair: {pair_id}")
    print("=" * 70)
    print(f"  website ({website_id}): {len(website_text)} chars")
    print(f"  second  ({vendor_id}):  {len(vendor_text)} chars "
          f"[extracted as {second_role}]")

    website_statements = extract_pps_from_policy(
        website_text, policy_source="first_party", policy_id=website_id,
    )
    vendor_statements = extract_pps_from_policy(
        vendor_text, policy_source=second_policy_source, policy_id=vendor_id,
    )

    website_gc = compute_gdpr_completeness(website_statements)
    vendor_gc  = compute_gdpr_completeness(vendor_statements)
    # Segment-level coverage + extraction gap, attached on both sides.
    # Cached after first run, so this is effectively free on reruns.
    website_clause_cov = compute_clause_gdpr_coverage(website_text, f"{pair_id}_website")
    vendor_clause_cov  = compute_clause_gdpr_coverage(vendor_text,  f"{pair_id}_vendor")
    website_gc["clause_coverage"] = website_clause_cov
    website_gc["extraction_gap"]  = compute_extraction_gap(website_gc, website_clause_cov)
    vendor_gc["clause_coverage"]  = vendor_clause_cov
    vendor_gc["extraction_gap"]   = compute_extraction_gap(vendor_gc, vendor_clause_cov)
    diff = compare_gdpr_completeness(website_gc, vendor_gc)

    elapsed = _time.monotonic() - t0
    result = {
        "pair_id": pair_id,
        "website_id": website_id,
        "vendor_id": vendor_id,
        "second_role": second_role,
        "timestamp": datetime.now().isoformat(),
        "reproducibility": {
            "commit_sha": _COMMIT_SHA,
            "ontology_hash": _ONTOLOGY_HASH,
            "patterns_version": _PATTERNS_VERSION,
            "extraction_prompt_version": EXTRACTION_PROMPT_VERSION,
            "scope_prompt_version": SCOPE_PROMPT_VERSION,
        },
        "n_website_statements": len(website_statements),
        "n_vendor_statements": len(vendor_statements),
        "gdpr_completeness": {"website": website_gc, "vendor": vendor_gc},
        "gdpr_comparison": diff,
        "elapsed_s": round(elapsed, 2),
    }

    out_path = output_path / f"{pair_id}_gdpr_comparison.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  GDPR pair comparison JSON: {out_path}")

    # Console summary — focus on the comparative signal.
    print(
        f"  Website PPS coverage: {website_gc['coverage_pct']:.1f}% | "
        f"Second PPS coverage: {vendor_gc['coverage_pct']:.1f}% | "
        f"Δ: {diff['coverage_pct_delta']:+.1f} pp"
    )
    w_clause_pct = float(website_clause_cov.get("coverage_pct", 0.0))
    v_clause_pct = float(vendor_clause_cov.get("coverage_pct", 0.0))
    print(
        f"  Segment-level:        website {w_clause_pct:.1f}% | "
        f"second {v_clause_pct:.1f}% (disclosure upper bound)"
    )
    w_gap = len(website_gc["extraction_gap"].get("extraction_gap_categories", []))
    v_gap = len(vendor_gc["extraction_gap"].get("extraction_gap_categories", []))
    if w_gap or v_gap:
        print(f"  Extraction gap:       website {w_gap} | second {v_gap}")
    print(
        f"  only_website_covers: {len(diff['only_website_covers'])} | "
        f"only_vendor_covers: {len(diff['only_vendor_covers'])} | "
        f"both_cover: {len(diff['both_cover'])} | "
        f"both_miss: {len(diff['both_miss'])}"
    )
    flags = diff["asymmetry_flags"]
    active = [name for name, on in flags.items() if on]
    if active:
        print(f"  Asymmetry flags: {', '.join(active)}")
    else:
        print("  Asymmetry flags: none")
    print(f"  Elapsed: {elapsed:.1f}s")
    return result


def run_gdpr_batch(
    manifest_path: str | Path,
    output_dir: str | Path | None = None,
) -> list[dict]:
    """Run GDPR-completeness on a CSV manifest of standalone policies.

    Manifest schema (flexible — first column taken as policy_id if present):
        policy_id,policy_file,policy_source
        my_policy_1,path/to/policy.txt,first_party
        vendor_a,vendors/a.txt,third_party
        vendor_b,vendors/b.txt,third_party

    ``policy_source`` defaults to ``first_party`` when absent. Writes one
    ``<policy_id>_gdpr_completeness.json`` per row plus an aggregate
    ``gdpr_completeness_summary.csv`` table (one row per policy with
    coverage_pct / is_complete / missing_count / per-family-pct) for the
    paper's cross-policy completeness figure.
    """
    manifest = Path(manifest_path)
    output_path = Path(output_dir) if output_dir else OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    with manifest.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    print(f"\nGDPR-completeness batch: {len(rows)} policies")

    results: list[dict] = []
    failures: list[dict] = []
    for i, row in enumerate(rows, start=1):
        policy_id = row.get("policy_id") or f"policy_{i}"
        policy_file_raw = (row.get("policy_file") or "").strip()
        if not policy_file_raw:
            failures.append({
                "row_index": i,
                "policy_id": policy_id,
                "error": "missing policy_file column value",
            })
            print(f"  ERROR {policy_id}: missing policy_file column value")
            continue
        policy_file = Path(policy_file_raw)
        policy_source = (row.get("policy_source") or "first_party").strip() or "first_party"
        if not policy_file.is_absolute():
            policy_file = manifest.parent / policy_file
        try:
            res = run_single_policy_gdpr(
                policy_file.read_text(encoding="utf-8"),
                policy_id=policy_id,
                policy_source=policy_source,
                output_dir=output_path,
            )
            results.append(res)
        except Exception as exc:
            failures.append({
                "row_index": i,
                "policy_id": policy_id,
                "policy_file": str(policy_file),
                "policy_source": policy_source,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            print(f"  ERROR {policy_id}: {exc}")

    # Aggregate CSV — one row per policy, easy to load into pandas / R.
    agg_csv = _write_gdpr_batch_summary_csv(results, output_path)
    print(f"\nAggregate CSV: {agg_csv}")

    # Failure ledger — only written when at least one row failed so that
    # successful runs don't leave a stale empty artifact on disk.
    if failures:
        failure_path = output_path / "gdpr_batch_failures.json"
        failure_path.write_text(
            json.dumps(
                {
                    "manifest": str(manifest),
                    "n_total": len(rows),
                    "n_succeeded": len(results),
                    "n_failed": len(failures),
                    "failures": failures,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"Failure ledger: {failure_path}  ({len(failures)}/{len(rows)} rows failed)")
    return results


def _write_gdpr_batch_summary_csv(
    results: list[dict],
    output_path: Path,
) -> Path:
    """Write the aggregate per-policy GDPR summary CSV.

    Pure data-shaping — no I/O beyond the CSV write, no LLM calls.
    Extracted from run_gdpr_batch so the aggregation logic is testable
    without standing up an LLM extraction backend.

    Each input dict is expected to match the shape returned by
    run_single_policy_gdpr(): {policy_id, policy_source,
    gdpr_completeness: {n_statements, coverage_pct, is_complete, missing,
    per_family_coverage: {Data Handling, Data Subject Rights,
    Legal / Organisational}}}.
    """
    agg_csv = output_path / "gdpr_completeness_summary.csv"
    with agg_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "policy_id", "policy_source", "n_statements", "coverage_pct",
            "is_complete", "missing_count",
            "data_handling_pct", "dsr_pct", "legal_pct",
            "missing_categories",
        ])
        for r in results:
            gc = r["gdpr_completeness"]
            fam = gc.get("per_family_coverage", {})
            writer.writerow([
                r["policy_id"],
                r["policy_source"],
                gc["n_statements"],
                gc["coverage_pct"],
                gc["is_complete"],
                len(gc["missing"]),
                fam.get("Data Handling", {}).get("coverage_pct", ""),
                fam.get("Data Subject Rights", {}).get("coverage_pct", ""),
                fam.get("Legal / Organisational", {}).get("coverage_pct", ""),
                "; ".join(gc["missing"]),
            ])
    return agg_csv


def run_demo(output_dir: str | Path | None = None) -> dict:
    """Run the built-in ExampleShop/Google Analytics demo pair."""

    print("\n" + "=" * 70)
    print("  PoliReasoner - Demo Mode")
    print("  Using built-in ExampleShop.com + Google Analytics policies")
    print("=" * 70)
    return run_pair(
        DEMO_WEBSITE_POLICY,
        DEMO_VENDOR_POLICY,
        vendor_name="Google Analytics",
        service_type="analytics",
        output_dir=output_dir,
        pair_id="demo",
    )
