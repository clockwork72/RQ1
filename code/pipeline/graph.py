"""Knowledge graph construction, merging, and graph-derived helpers."""

from __future__ import annotations

import re

import networkx as nx

from .normalizer import (
    CANONICAL_DATA_TYPES,
    DATA_ONTOLOGY,
    data_subsumes,
    data_types_related,
    normalize_actor,
    normalize_data_type,
)
from .schema import Modality, PPS


PROCESSING_ACTIONS = {"collect", "use", "share", "sell", "retain", "transfer", "process"}


def _safe_label(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip())
    return value or "unknown"


def _practice_edge_type(pps: PPS) -> str:
    if pps.is_negative or pps.modality == Modality.PROHIBITION:
        return f"NOT_{pps.action.upper()}"
    return pps.action.upper()


def build_graph(
    statements: list[PPS],
    policy_label: str,
    first_party_names: list[str] | None = None,
) -> nx.MultiDiGraph:
    """Build a policy graph with ontology, entities, data nodes, and PPS edges.

    Parameters
    ----------
    first_party_names : list[str] | None
        Company names to treat as first party during actor normalization
        (e.g., ["Sedo", "Sedo.com LLC"]).
    """

    graph = nx.MultiDiGraph(policy_label=policy_label)

    # Ontology nodes are always canonical by construction.
    for parent, children in DATA_ONTOLOGY.items():
        parent_id = f"data:{parent}"
        graph.add_node(parent_id, label=parent, node_type="DataType",
                       is_canonical=True, policy_label=policy_label)
        for child in children:
            child_id = f"data:{child}"
            graph.add_node(child_id, label=child, node_type="DataType",
                           is_canonical=True, policy_label=policy_label)
            graph.add_edge(
                parent_id,
                child_id,
                key=f"subsumes:{parent}->{child}",
                edge_type="SUBSUME",
                policy_label=policy_label,
            )

    for statement in statements:
        actor_normalized = normalize_actor(statement.actor, first_party_names)
        actor_id = f"entity:{_safe_label(actor_normalized)}"
        data_label = normalize_data_type(statement.data_object)
        data_id = f"data:{data_label}"
        # Flag data-type nodes that came from passthrough (extractor emitted a
        # term the normalizer couldn't canonicalize). Downstream ALIGNED_TO
        # generation skips these so extractor hallucinations can't produce
        # cross-policy alignment edges.
        is_canonical = data_label in CANONICAL_DATA_TYPES

        graph.add_node(
            actor_id,
            label=actor_normalized,
            node_type="Entity",
            policy_source=statement.policy_source,
            policy_label=policy_label,
        )
        graph.add_node(
            data_id,
            label=data_label,
            node_type="DataType",
            is_canonical=is_canonical,
            policy_source=statement.policy_source,
            policy_label=policy_label,
        )

        edge_attrs = {
            "edge_type": _practice_edge_type(statement),
            "pps_id": statement.id,
            "action": statement.action,
            "modality": statement.modality.name,
            "purpose": statement.purpose,
            "recipient": statement.recipient,
            "condition": statement.condition.value,
            "temporality": statement.temporality.value,
            "temporality_value": statement.temporality_value,
            "is_negative": statement.is_negative,
            "source_text": statement.source_text,
            "source_section": statement.source_section,
            "policy_source": statement.policy_source,
            "policy_label": policy_label,
            "gdpr_categories": list(statement.gdpr_categories),
            "pps": statement.to_dict(),
        }
        graph.add_edge(actor_id, data_id, key=statement.id, **edge_attrs)

        if statement.recipient and statement.action in {"share", "sell", "transfer"}:
            recipient_normalized = normalize_actor(statement.recipient)
            recipient_id = f"entity:{_safe_label(recipient_normalized)}"
            graph.add_node(
                recipient_id,
                label=recipient_normalized,
                node_type="Entity",
                policy_source=statement.policy_source,
                policy_label=policy_label,
            )
            graph.add_edge(
                actor_id,
                recipient_id,
                key=f"{statement.id}:recipient",
                edge_type="SHARES_WITH",
                pps_id=statement.id,
                action=statement.action,
                modality=statement.modality.name,
                purpose=statement.purpose,
                recipient=recipient_normalized,
                condition=statement.condition.value,
                data_object=data_label,
                is_negative=statement.is_negative,
                source_text=statement.source_text,
                source_section=statement.source_section,
                policy_source=statement.policy_source,
                policy_label=policy_label,
                gdpr_categories=list(statement.gdpr_categories),
                pps=statement.to_dict(),
            )

    print(
        f"  Graph '{policy_label}': {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
    )
    return graph


def _copy_into_merged(
    target: nx.MultiDiGraph,
    source: nx.MultiDiGraph,
    prefix: str,
    graph_source: str,
) -> None:
    for node_id, attrs in source.nodes(data=True):
        target.add_node(f"{prefix}:{node_id}", **attrs, graph_source=graph_source)

    for source_u, source_v, key, attrs in source.edges(keys=True, data=True):
        target.add_edge(
            f"{prefix}:{source_u}",
            f"{prefix}:{source_v}",
            key=f"{prefix}:{key}",
            **attrs,
            graph_source=graph_source,
        )


def _data_nodes_with_practices(graph: nx.MultiDiGraph, prefix: str) -> dict[str, str]:
    nodes: dict[str, str] = {}
    for _, target_node, attrs in graph.edges(data=True):
        if target_node.startswith(f"{prefix}:data:") and attrs.get("pps"):
            label = graph.nodes[target_node].get("label", "")
            nodes[label] = target_node
    return nodes


def merge_graphs(
    g_website: nx.MultiDiGraph,
    g_vendor: nx.MultiDiGraph,
    vendor_name: str,
    service_type: str = "analytics",
) -> nx.MultiDiGraph:
    """Merge first-party and vendor graphs with bridge and alignment edges."""

    merged = nx.MultiDiGraph(service_type=service_type, vendor_name=vendor_name)
    _copy_into_merged(merged, g_website, "1p", "website")
    _copy_into_merged(merged, g_vendor, "3p", "vendor")

    bridge_id = f"bridge:{vendor_name}"
    merged.add_node(
        bridge_id,
        label=vendor_name,
        node_type="Bridge",
        graph_source="merged",
        service_type=service_type,
    )

    for source_u, source_v, key, attrs in list(merged.edges(keys=True, data=True)):
        # Only emit SENDS_DATA_VIA from first-party actor→data practice edges.
        # Recipient (SHARES_WITH) edges target entity nodes — iterating them
        # here produced duplicate bridge edges whose data_object was actually
        # the recipient's name.
        if not source_u.startswith("1p:entity:"):
            continue
        if not source_v.startswith("1p:data:"):
            continue
        if attrs.get("edge_type") == "SHARES_WITH":
            continue
        if attrs.get("action") not in {"share", "sell", "transfer"}:
            continue
        if attrs.get("is_negative"):
            continue
        # Recipient-gated bridge: the website practice must actually name
        # this vendor (or have no recipient and fall back to the type-only
        # match done by downstream get_bridge_flows / Π₈ alignment). This
        # stops "share with advertising partners" from confirming a bridge
        # to e.g. Google Analytics.
        recipient_text = (attrs.get("recipient") or "").strip().lower()
        bridge_confirmed = (
            not recipient_text
            or vendor_name.lower() in recipient_text
            or recipient_text in vendor_name.lower()
        )
        merged.add_edge(
            source_u,
            bridge_id,
            key=f"{key}:bridge",
            edge_type="SENDS_DATA_VIA",
            pps_id=attrs.get("pps_id", ""),
            # G2: propagate the originating PPS dict so
            # extract_statements_from_graph can recover the statement when
            # iterating bridge edges. Without this, bridge edges were skipped
            # on round-trip because they lacked the `pps` attribute.
            pps=attrs.get("pps"),
            originating_edge=key,
            action=attrs.get("action", ""),
            modality=attrs.get("modality", "UNSPECIFIED"),
            data_object=merged.nodes.get(source_v, {}).get("label", attrs.get("data_object", "")),
            purpose=attrs.get("purpose", ""),
            condition=attrs.get("condition", "unspecified"),
            recipient=attrs.get("recipient", ""),
            bridge_confirmed=bridge_confirmed,
            source_text=attrs.get("source_text", ""),
            policy_source=attrs.get("policy_source", ""),
            graph_source="merged",
        )

    for source_u, source_v, key, attrs in list(merged.edges(keys=True, data=True)):
        if not source_u.startswith("3p:entity:"):
            continue
        if attrs.get("action") not in PROCESSING_ACTIONS:
            continue
        if attrs.get("is_negative"):
            continue
        if not source_v.startswith("3p:data:"):
            continue
        merged.add_edge(
            bridge_id,
            source_v,
            key=f"{key}:bridge_in",
            edge_type="DATA_PROCESSED_BY",
            pps_id=attrs.get("pps_id", ""),
            # G2: see SENDS_DATA_VIA rationale above.
            pps=attrs.get("pps"),
            originating_edge=key,
            action=attrs.get("action", ""),
            modality=attrs.get("modality", "UNSPECIFIED"),
            data_object=merged.nodes[source_v].get("label", ""),
            purpose=attrs.get("purpose", ""),
            condition=attrs.get("condition", "unspecified"),
            temporality=attrs.get("temporality", "unspecified"),
            temporality_value=attrs.get("temporality_value", ""),
            source_text=attrs.get("source_text", ""),
            policy_source=attrs.get("policy_source", ""),
            graph_source="merged",
        )

    website_data = _data_nodes_with_practices(merged, "1p")
    vendor_data = _data_nodes_with_practices(merged, "3p")
    for website_label, website_node in website_data.items():
        # G1: uncanonical data nodes (extractor passthrough — e.g., novel
        # research jargon not in the ontology) must not participate in
        # ALIGNED_TO. Default True for backward compat with older graphs
        # that pre-date the is_canonical flag.
        if not merged.nodes[website_node].get("is_canonical", True):
            continue
        for vendor_label, vendor_node in vendor_data.items():
            if not merged.nodes[vendor_node].get("is_canonical", True):
                continue
            # Round 9 (sibling-FP fix): only align when the data types are
            # identical or one strictly subsumes the other. Sibling data
            # types (e.g., "ip address" vs "device id") must not produce
            # ALIGNED_TO edges because downstream graph-aware Π₈ trusts the
            # alignment as if it were exact (skip_data_match=True) and fires
            # spurious temporal contradictions across distinct data types.
            w_norm = normalize_data_type(website_label)
            v_norm = normalize_data_type(vendor_label)
            if not (w_norm and v_norm):
                continue
            if w_norm == v_norm:
                relation = "exact"
            elif data_subsumes(w_norm, v_norm) or data_subsumes(v_norm, w_norm):
                relation = "related"
            else:
                continue
            merged.add_edge(
                website_node,
                vendor_node,
                key=f"align:{website_label}->{vendor_label}",
                edge_type="ALIGNED_TO",
                relation=relation,
                graph_source="merged",
            )

    print(
        f"  Merged graph: {merged.number_of_nodes()} nodes, {merged.number_of_edges()} edges"
    )
    return merged


def compute_graph_metrics(graph: nx.MultiDiGraph) -> dict:
    """Compute summary metrics for a graph."""

    node_types: dict[str, int] = {}
    for _, attrs in graph.nodes(data=True):
        node_type = attrs.get("node_type", "Unknown")
        node_types[node_type] = node_types.get(node_type, 0) + 1

    edge_types: dict[str, int] = {}
    for _, _, attrs in graph.edges(data=True):
        edge_type = attrs.get("edge_type", "Unknown")
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

    simple_graph = nx.DiGraph()
    simple_graph.add_nodes_from(graph.nodes(data=True))
    for source_u, source_v in graph.edges():
        simple_graph.add_edge(source_u, source_v)

    return {
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
        "node_types": node_types,
        "edge_types": edge_types,
        "density": nx.density(simple_graph),
        "weakly_connected_components": nx.number_weakly_connected_components(simple_graph),
    }


def extract_statements_from_graph(
    graph: nx.MultiDiGraph,
    graph_source: str | None = None,
) -> list[PPS]:
    """Reconstruct PPS objects stored on graph edges."""

    statements: dict[str, PPS] = {}
    for _, _, attrs in graph.edges(data=True):
        if graph_source and attrs.get("graph_source") != graph_source:
            continue
        raw_pps = attrs.get("pps")
        if not raw_pps:
            continue
        statement = PPS.from_dict(raw_pps)
        statements.setdefault(statement.id, statement)
    return list(statements.values())
