"""Graph-neighborhood accessors for structure-aware pattern matching.

Provides DataNeighborhood and AlignedNeighborhoodPair abstractions that let
patterns operate on data-type clusters instead of flat PPS lists. Each
neighborhood groups all PPS edges incident on a data-type node (and its
ontology descendants) within a single policy source.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from .normalizer import DATA_ONTOLOGY, data_subsumes, normalize_data_type
from .schema import PPS


@dataclass
class DataNeighborhood:
    """All PPS statements incident on a data-type node cluster."""

    data_type: str
    data_node_ids: list[str] = field(default_factory=list)
    statements: list[PPS] = field(default_factory=list)
    policy_source: str = ""


@dataclass
class AlignedNeighborhoodPair:
    """A website neighborhood aligned to a vendor neighborhood via ALIGNED_TO."""

    website: DataNeighborhood
    vendor: DataNeighborhood
    alignment_relation: str = "related"


@dataclass
class BridgeFlow:
    """A confirmed data handoff: website shares via bridge, vendor processes."""

    website_pps: PPS
    vendor_pps: PPS
    data_type: str
    bridge_node: str


def _get_ontology_descendants(canonical_type: str) -> set[str]:
    """Return all descendant types (including self) from DATA_ONTOLOGY."""
    result = {canonical_type}
    children = DATA_ONTOLOGY.get(canonical_type, [])
    for child in children:
        result |= _get_ontology_descendants(child)
    return result


def _get_ontology_ancestors(canonical_type: str) -> set[str]:
    """Return all ancestor types (excluding self) from DATA_ONTOLOGY."""
    ancestors: set[str] = set()
    for parent, children in DATA_ONTOLOGY.items():
        if canonical_type in children:
            ancestors.add(parent)
            ancestors |= _get_ontology_ancestors(parent)
    return ancestors


def _pps_from_edge(attrs: dict) -> PPS | None:
    """Reconstruct a PPS from edge attributes, or None if no PPS stored."""
    raw = attrs.get("pps")
    if not raw:
        return None
    return PPS.from_dict(raw)


def get_data_neighborhoods(
    graph: nx.MultiDiGraph,
    policy_prefix: str,
    expand_ancestors: bool = False,
) -> list[DataNeighborhood]:
    """Extract data-type neighborhoods from a merged graph.

    Parameters
    ----------
    graph : merged MultiDiGraph from merge_graphs()
    policy_prefix : "1p" for website, "3p" for vendor
    expand_ancestors : if True, parent neighborhoods include descendant
        statements. Set False for intra-policy patterns (avoids duplicate
        comparisons across ontology levels).
    """
    prefix = f"{policy_prefix}:data:"
    policy_source = "website" if policy_prefix == "1p" else "vendor"

    node_to_canonical: dict[str, str] = {}
    for node_id, attrs in graph.nodes(data=True):
        if not node_id.startswith(prefix):
            continue
        if attrs.get("node_type") != "DataType":
            continue
        label = attrs.get("label", "")
        canonical = normalize_data_type(label) or label.lower().strip()
        node_to_canonical[node_id] = canonical

    canonical_to_nodes: dict[str, set[str]] = {}
    for node_id, canonical in node_to_canonical.items():
        canonical_to_nodes.setdefault(canonical, set()).add(node_id)
        if expand_ancestors:
            for ancestor in _get_ontology_ancestors(canonical):
                canonical_to_nodes.setdefault(ancestor, set()).add(node_id)

    neighborhoods: list[DataNeighborhood] = []
    for canonical, node_ids in canonical_to_nodes.items():
        statements: dict[str, PPS] = {}
        for node_id in node_ids:
            for source, _, attrs in graph.in_edges(node_id, data=True):
                pps = _pps_from_edge(attrs)
                if pps:
                    statements.setdefault(pps.id, pps)
        if not statements:
            continue
        neighborhoods.append(
            DataNeighborhood(
                data_type=canonical,
                data_node_ids=sorted(node_ids),
                statements=list(statements.values()),
                policy_source=policy_source,
            )
        )

    return neighborhoods


def _get_subsume_descendants_in_graph(
    graph: nx.MultiDiGraph,
    node_id: str,
) -> set[str]:
    """Walk SUBSUME edges downward from a node, returning all descendant node IDs."""
    descendants: set[str] = set()
    for _, child, attrs in graph.out_edges(node_id, data=True):
        if attrs.get("edge_type") == "SUBSUME":
            descendants.add(child)
            descendants |= _get_subsume_descendants_in_graph(graph, child)
    return descendants


def _is_ontology_leaf(canonical_type: str) -> bool:
    """True if this type has no children in DATA_ONTOLOGY."""
    return canonical_type not in DATA_ONTOLOGY or not DATA_ONTOLOGY[canonical_type]


def get_aligned_pairs(
    graph: nx.MultiDiGraph,
) -> list[AlignedNeighborhoodPair]:
    """Find cross-policy aligned neighborhood pairs via ALIGNED_TO edges.

    Uses SUBSUME edges to expand neighborhoods with descendant statements,
    then applies leaf-first deduplication: if a specific child type (e.g.
    "email address") has its own aligned pair on both sides, those statements
    are removed from the parent pair (e.g. "personal data") to avoid the
    mega-neighborhood problem.

    Returns one AlignedNeighborhoodPair per aligned data-type cluster.
    Only includes pairs where both sides have >=1 PPS statement.
    """
    alignments: dict[str, dict] = {}

    for u, v, attrs in graph.edges(data=True):
        if attrs.get("edge_type") != "ALIGNED_TO":
            continue
        relation = attrs.get("relation", "related")

        website_node = u if u.startswith("1p:") else v
        vendor_node = v if v.startswith("3p:") else u
        if not website_node.startswith("1p:") or not vendor_node.startswith("3p:"):
            continue

        w_label = graph.nodes[website_node].get("label", "")
        v_label = graph.nodes[vendor_node].get("label", "")
        w_canonical = normalize_data_type(w_label) or w_label.lower().strip()
        v_canonical = normalize_data_type(v_label) or v_label.lower().strip()
        key = f"{w_canonical}||{v_canonical}"

        if key not in alignments:
            alignments[key] = {
                "w_canonical": w_canonical,
                "v_canonical": v_canonical,
                "w_nodes": set(),
                "v_nodes": set(),
                "relation": relation,
            }
        alignments[key]["w_nodes"].add(website_node)
        alignments[key]["v_nodes"].add(vendor_node)
        if relation == "exact":
            alignments[key]["relation"] = "exact"

    same_type_keys: set[str] = set()
    for key, info in alignments.items():
        if info["w_canonical"] == info["v_canonical"]:
            same_type_keys.add(key)

    # SUBSUME expansion: only for same-type alignments.
    # Cross-type pairs use direct nodes only to avoid mega-neighborhood explosion.
    for key in same_type_keys:
        info = alignments[key]
        w_expanded: set[str] = set()
        for wn in list(info["w_nodes"]):
            w_expanded |= _get_subsume_descendants_in_graph(graph, wn)
        v_expanded: set[str] = set()
        for vn in list(info["v_nodes"]):
            v_expanded |= _get_subsume_descendants_in_graph(graph, vn)
        info["w_nodes_expanded"] = info["w_nodes"] | w_expanded
        info["v_nodes_expanded"] = info["v_nodes"] | v_expanded

    # Cross-type alignments: no expansion
    for key, info in alignments.items():
        if key not in same_type_keys:
            info["w_nodes_expanded"] = info["w_nodes"]
            info["v_nodes_expanded"] = info["v_nodes"]

    same_type_canonicals = {
        alignments[k]["w_canonical"] for k in same_type_keys
    }

    child_claimed_w: set[str] = set()
    child_claimed_v: set[str] = set()

    specificity_order = sorted(
        same_type_keys,
        key=lambda k: len(_get_ontology_descendants(alignments[k]["w_canonical"])),
    )

    results: list[AlignedNeighborhoodPair] = []

    for key in specificity_order:
        info = alignments[key]
        canonical = info["w_canonical"]

        w_stmts = _collect_statements_for_nodes(graph, info["w_nodes_expanded"])
        v_stmts = _collect_statements_for_nodes(graph, info["v_nodes_expanded"])

        # Always skip statements already claimed by a more-specific ancestor
        # pair. Previously intermediates (e.g., "contact information") did
        # not claim their own statements when they had further descendants,
        # so those PPS leaked into every ancestor pair via SUBSUME expansion
        # and produced duplicate Π₈ findings (w_ci/v_ci and w_ci/v_pd).
        w_stmts = [s for s in w_stmts if s.id not in child_claimed_w]
        v_stmts = [s for s in v_stmts if s.id not in child_claimed_v]

        if not w_stmts or not v_stmts:
            continue

        child_claimed_w.update(s.id for s in w_stmts)
        child_claimed_v.update(s.id for s in v_stmts)

        results.append(
            AlignedNeighborhoodPair(
                website=DataNeighborhood(
                    data_type=canonical,
                    data_node_ids=sorted(info["w_nodes_expanded"]),
                    statements=w_stmts,
                    policy_source="website",
                ),
                vendor=DataNeighborhood(
                    data_type=canonical,
                    data_node_ids=sorted(info["v_nodes_expanded"]),
                    statements=v_stmts,
                    policy_source="vendor",
                ),
                alignment_relation=info["relation"],
            )
        )

    # Cross-type alignments: by default skipped because same-type pairs with
    # SUBSUME expansion cover the same comparisons. HOWEVER, SUBSUME expands
    # only to descendants, so a broad parent-side PPS (e.g., website
    # prohibition on "personal data") is NOT picked up by the child
    # "email ↔ email" same-type alignment — that alignment's expanded set
    # includes email's descendants, not its ancestors. Keep the cross-type
    # pair when the parent-side direct PPS are absent from every same-type
    # alignment's statement set, otherwise the broad parent evidence is
    # silently dropped from graph-aware Π₈.
    same_type_stmt_ids: set[str] = set()
    for key in same_type_keys:
        info = alignments[key]
        for pps in _collect_statements_for_nodes(graph, info["w_nodes_expanded"]):
            same_type_stmt_ids.add(pps.id)
        for pps in _collect_statements_for_nodes(graph, info["v_nodes_expanded"]):
            same_type_stmt_ids.add(pps.id)

    for key, info in alignments.items():
        if key in same_type_keys:
            continue
        w_covered = info["w_canonical"] in same_type_canonicals or any(
            data_subsumes(stc, info["w_canonical"]) or data_subsumes(info["w_canonical"], stc)
            for stc in same_type_canonicals
        )
        v_covered = info["v_canonical"] in same_type_canonicals or any(
            data_subsumes(stc, info["v_canonical"]) or data_subsumes(info["v_canonical"], stc)
            for stc in same_type_canonicals
        )
        if w_covered and v_covered:
            # Rescue broad-parent cross-type pairs: when the parent side (the
            # ontology ancestor) has direct PPS not represented in any
            # same-type alignment, keep the cross-type pair so those PPS can
            # still match the child-side vendor practices.
            w_direct = _collect_statements_for_nodes(graph, info["w_nodes"])
            v_direct = _collect_statements_for_nodes(graph, info["v_nodes"])
            w_ancestor = data_subsumes(info["w_canonical"], info["v_canonical"])
            v_ancestor = data_subsumes(info["v_canonical"], info["w_canonical"])
            parent_has_orphan = False
            if w_ancestor and any(s.id not in same_type_stmt_ids for s in w_direct):
                parent_has_orphan = True
            if v_ancestor and any(s.id not in same_type_stmt_ids for s in v_direct):
                parent_has_orphan = True
            if not parent_has_orphan:
                continue
        w_stmts = _collect_statements_for_nodes(graph, info["w_nodes_expanded"])
        v_stmts = _collect_statements_for_nodes(graph, info["v_nodes_expanded"])
        if not w_stmts or not v_stmts:
            continue
        results.append(
            AlignedNeighborhoodPair(
                website=DataNeighborhood(
                    data_type=info["w_canonical"],
                    data_node_ids=sorted(info["w_nodes_expanded"]),
                    statements=w_stmts,
                    policy_source="website",
                ),
                vendor=DataNeighborhood(
                    data_type=info["v_canonical"],
                    data_node_ids=sorted(info["v_nodes_expanded"]),
                    statements=v_stmts,
                    policy_source="vendor",
                ),
                alignment_relation=info["relation"],
            )
        )

    return results


def _has_child_alignment(
    canonical: str,
    same_type_canonicals: set[str],
) -> bool:
    """Check if any child type in the ontology also has a same-type alignment."""
    children = DATA_ONTOLOGY.get(canonical, [])
    for child in children:
        if child in same_type_canonicals:
            return True
        if _has_child_alignment(child, same_type_canonicals):
            return True
    return False


def _collect_statements_for_nodes(
    graph: nx.MultiDiGraph,
    node_ids: set[str],
) -> list[PPS]:
    """Collect all PPS from edges targeting the given data-type nodes."""
    statements: dict[str, PPS] = {}
    for node_id in node_ids:
        for source, _, attrs in graph.in_edges(node_id, data=True):
            pps = _pps_from_edge(attrs)
            if pps:
                statements.setdefault(pps.id, pps)
    return list(statements.values())


def get_bridge_flows(
    graph: nx.MultiDiGraph,
    vendor_name: str,
) -> list[BridgeFlow]:
    """Trace SENDS_DATA_VIA -> bridge -> DATA_PROCESSED_BY paths.

    Each flow represents a confirmed data handoff where the website shares
    data via a bridge node and the vendor processes it on the other side.
    """
    bridge_id = f"bridge:{vendor_name}"
    if bridge_id not in graph:
        return []

    incoming: dict[str, list[dict]] = {}
    outgoing: dict[str, list[dict]] = {}

    for u, v, attrs in graph.edges(data=True):
        if v == bridge_id and attrs.get("edge_type") == "SENDS_DATA_VIA":
            # Recipient-gated bridges: only count the handoff when the
            # website practice actually names this vendor (bridge_confirmed
            # set by merge_graphs). An edge with bridge_confirmed=False came
            # from a "share only with <other vendor>" clause and must not
            # confirm a bridge to the current vendor.
            if not attrs.get("bridge_confirmed", True):
                continue
            data_obj = attrs.get("data_object", "")
            canonical = normalize_data_type(data_obj) or data_obj.lower().strip()
            incoming.setdefault(canonical, []).append(attrs)
        elif u == bridge_id and attrs.get("edge_type") == "DATA_PROCESSED_BY":
            data_obj = attrs.get("data_object", "")
            canonical = normalize_data_type(data_obj) or data_obj.lower().strip()
            outgoing.setdefault(canonical, []).append(attrs)

    edge_index = _build_edge_index(graph)
    flows: list[BridgeFlow] = []
    seen_pairs: set[tuple[str, str, str]] = set()
    out_types = list(outgoing.keys())
    for in_type, in_edges in incoming.items():
        # Bridge evidence is confirmed whenever the website's shared data type
        # is equal to, subsumes, or is subsumed by the vendor's processed type.
        # Exact-only matching silently dropped broad→specific handoffs
        # (e.g., website shares "personal data", vendor processes "email").
        matched_out_types = [
            out_type for out_type in out_types
            if out_type == in_type
            or data_subsumes(in_type, out_type)
            or data_subsumes(out_type, in_type)
        ]
        for out_type in matched_out_types:
            out_edges = outgoing.get(out_type, [])
            if not out_edges:
                continue
            for in_edge in in_edges:
                in_pps = _pps_from_originating_edge(edge_index, in_edge)
                if not in_pps:
                    continue
                for out_edge in out_edges:
                    out_pps = _pps_from_originating_edge(edge_index, out_edge)
                    if not out_pps:
                        continue
                    flow_key = (in_pps.id, out_pps.id, in_type)
                    if flow_key in seen_pairs:
                        continue
                    seen_pairs.add(flow_key)
                    flows.append(
                        BridgeFlow(
                            website_pps=in_pps,
                            vendor_pps=out_pps,
                            data_type=in_type,
                            bridge_node=bridge_id,
                        )
                    )

    return flows


def _build_edge_index(graph: nx.MultiDiGraph) -> dict[str, dict]:
    """Build a {key: attrs} index for O(1) edge lookup."""
    index: dict[str, dict] = {}
    for u, v, key, attrs in graph.edges(keys=True, data=True):
        index[key] = attrs
    return index


def _pps_from_originating_edge(edge_index: dict[str, dict], bridge_attrs: dict) -> PPS | None:
    """Recover the original PPS from a bridge edge's originating_edge reference."""
    orig_key = bridge_attrs.get("originating_edge", "")
    if not orig_key:
        return None
    attrs = edge_index.get(orig_key)
    if attrs is None:
        return None
    return _pps_from_edge(attrs)


def get_neighborhood_context(neighborhood: DataNeighborhood) -> dict:
    """Summarize a neighborhood for cluster-level pre-filtering."""
    from schema import Modality, TemporalityType
    from normalizer import normalize_purpose

    modalities: set[str] = set()
    actions: set[str] = set()
    purposes: set[str] = set()
    conditions: dict[str, int] = {}
    has_prohibition = False
    has_exclusivity = False
    has_retention = False

    for s in neighborhood.statements:
        modalities.add(s.modality.name)
        actions.add(s.action)
        if s.purpose:
            p = normalize_purpose(s.purpose)
            if p:
                purposes.add(p)
        cond = s.condition.value
        conditions[cond] = conditions.get(cond, 0) + 1
        if s.modality == Modality.PROHIBITION or s.is_negative:
            has_prohibition = True
        if s.temporality != TemporalityType.UNSPECIFIED:
            has_retention = True
        if "only " in s.source_text.lower() or "solely " in s.source_text.lower():
            has_exclusivity = True

    return {
        "data_type": neighborhood.data_type,
        "n_statements": len(neighborhood.statements),
        "modalities": modalities,
        "actions": actions,
        "purposes": purposes,
        "conditions": conditions,
        "has_prohibition": has_prohibition,
        "has_exclusivity": has_exclusivity,
        "has_retention": has_retention,
    }
