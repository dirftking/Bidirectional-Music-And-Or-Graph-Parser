"""Structural and semantic validation for Audio2MusicAOG candidate graphs."""

from __future__ import annotations

from collections import defaultdict
from math import isclose
from typing import Any


SCHEMA_VERSION = "0.2"
REPRESENTATION = "audio2musicaog_candidate_set"
NODE_KINDS = {"and", "or", "terminal"}
NODE_LEVELS = {"song", "section", "phrase", "metrical", "harmony", "note", "event"}
NODE_TYPES = {
    "song",
    "section",
    "section_label",
    "phrase",
    "measure",
    "beat",
    "chord",
    "note",
    "key_event",
    "instrument_event",
    "vocal_event",
}
RELATION_TYPES = {"succession", "variation", "diad", "synchronization"}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_confidence(value: Any, path: str, errors: list[str]) -> None:
    if not _is_number(value) or not 0 <= value <= 1:
        errors.append(f"{path} must be a number in [0, 1]")


def _cycle_nodes(adjacency: dict[str, list[str]], root: str) -> set[str]:
    visiting: set[str] = set()
    visited: set[str] = set()
    cycles: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            cycles.add(node)
            return
        if node in visited:
            return
        visiting.add(node)
        for child in adjacency.get(node, []):
            visit(child)
        visiting.remove(node)
        visited.add(node)

    visit(root)
    return cycles


def validate_graph(graph: dict[str, Any]) -> list[str]:
    """Return human-readable errors; an empty list means the graph is valid.

    JSON Schema checks field shapes.  This function additionally checks graph
    invariants that JSON Schema cannot conveniently express: references,
    acyclicity, reachability, temporal containment, and Or-choice probabilities.
    """

    errors: list[str] = []
    required_top = {
        "schema_version",
        "representation",
        "root",
        "nodes",
        "productions",
        "relations",
        "uncertainties",
        "provenance",
    }
    missing = sorted(required_top - set(graph))
    errors.extend(f"missing top-level field: {field}" for field in missing)
    if graph.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")
    if graph.get("representation") != REPRESENTATION:
        errors.append(f"representation must be {REPRESENTATION!r}")

    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return [*errors, "nodes must be an array"]

    node_by_id: dict[str, dict[str, Any]] = {}
    node_required = {"id", "kind", "level", "type", "span", "attributes", "confidence", "evidence"}
    for index, node in enumerate(nodes):
        path = f"nodes[{index}]"
        if not isinstance(node, dict):
            errors.append(f"{path} must be an object")
            continue
        for field in sorted(node_required - set(node)):
            errors.append(f"{path} missing field: {field}")
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            errors.append(f"{path}.id must be a non-empty string")
            continue
        if node_id in node_by_id:
            errors.append(f"duplicate node id: {node_id}")
        node_by_id[node_id] = node
        if node.get("kind") not in NODE_KINDS:
            errors.append(f"{path}.kind is invalid: {node.get('kind')!r}")
        if node.get("level") not in NODE_LEVELS:
            errors.append(f"{path}.level is invalid: {node.get('level')!r}")
        if node.get("type") not in NODE_TYPES:
            errors.append(f"{path}.type is invalid: {node.get('type')!r}")
        _check_confidence(node.get("confidence"), f"{path}.confidence", errors)
        if not isinstance(node.get("attributes"), dict):
            errors.append(f"{path}.attributes must be an object")
        if not isinstance(node.get("evidence"), list):
            errors.append(f"{path}.evidence must be an array")

        span = node.get("span")
        if not isinstance(span, dict):
            errors.append(f"{path}.span must be an object")
            continue
        start, end = span.get("start_seconds"), span.get("end_seconds")
        if not _is_number(start) or not _is_number(end) or start < 0 or end < start:
            errors.append(f"{path}.span must satisfy 0 <= start_seconds <= end_seconds")
        elif node.get("type") in {"song", "section", "section_label", "chord"} and end <= start:
            errors.append(f"{path}.span must have positive duration for node type {node.get('type')!r}")

    root = graph.get("root")
    if root not in node_by_id:
        errors.append(f"root references missing node: {root!r}")

    productions = graph.get("productions")
    if not isinstance(productions, list):
        productions = []
        errors.append("productions must be an array")
    production_ids: set[str] = set()
    adjacency: dict[str, list[str]] = defaultdict(list)
    incoming: dict[str, int] = defaultdict(int)
    productions_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, production in enumerate(productions):
        path = f"productions[{index}]"
        if not isinstance(production, dict):
            errors.append(f"{path} must be an object")
            continue
        production_id = production.get("id")
        if not isinstance(production_id, str) or not production_id:
            errors.append(f"{path}.id must be a non-empty string")
        elif production_id in production_ids:
            errors.append(f"duplicate production id: {production_id}")
        else:
            production_ids.add(production_id)
        parent = production.get("parent")
        children = production.get("children")
        if parent not in node_by_id:
            errors.append(f"{path}.parent references missing node: {parent!r}")
        if not isinstance(children, list) or not children:
            errors.append(f"{path}.children must be a non-empty array")
            children = []
        for child in children:
            if child not in node_by_id:
                errors.append(f"{path}.children references missing node: {child!r}")
                continue
            adjacency[parent].append(child)
            incoming[child] += 1
        _check_confidence(production.get("probability"), f"{path}.probability", errors)
        if not isinstance(production.get("selected"), bool):
            errors.append(f"{path}.selected must be boolean")
        if parent in node_by_id:
            productions_by_parent[parent].append(production)

    if root in node_by_id and incoming.get(root, 0):
        errors.append("root node must not be a production child")
    if root in node_by_id:
        cycles = _cycle_nodes(adjacency, root)
        if cycles:
            errors.append(f"production graph contains a cycle involving: {sorted(cycles)}")
        reachable: set[str] = set()
        stack = [root]
        while stack:
            current = stack.pop()
            if current in reachable:
                continue
            reachable.add(current)
            stack.extend(adjacency.get(current, []))
        unreachable = sorted(set(node_by_id) - reachable)
        if unreachable:
            errors.append(f"nodes unreachable from root: {unreachable}")

    for parent, choices in productions_by_parent.items():
        parent_kind = node_by_id[parent].get("kind")
        if parent_kind == "terminal":
            errors.append(f"terminal node has productions: {parent}")
        if parent_kind == "and" and len(choices) != 1:
            errors.append(f"and node must have exactly one production: {parent}")
        if parent_kind == "or":
            total = sum(item.get("probability", 0) for item in choices if _is_number(item.get("probability")))
            if not isclose(total, 1.0, abs_tol=1e-6):
                errors.append(f"or-node production probabilities must sum to 1: {parent} has {total:.6f}")
            if sum(item.get("selected") is True for item in choices) != 1:
                errors.append(f"or node must select exactly one production: {parent}")

    # Production children should be temporally contained by their parent.  An
    # epsilon tolerates rounded timestamps emitted by language models.
    for parent, children in adjacency.items():
        parent_span = node_by_id.get(parent, {}).get("span", {})
        p_start, p_end = parent_span.get("start_seconds"), parent_span.get("end_seconds")
        if not _is_number(p_start) or not _is_number(p_end):
            continue
        for child in children:
            child_span = node_by_id[child].get("span", {})
            c_start, c_end = child_span.get("start_seconds"), child_span.get("end_seconds")
            if _is_number(c_start) and _is_number(c_end) and (c_start < p_start - 1e-3 or c_end > p_end + 1e-3):
                errors.append(f"child span is outside parent span: {parent} -> {child}")

    relations = graph.get("relations")
    if not isinstance(relations, list):
        errors.append("relations must be an array")
        relations = []
    relation_ids: set[str] = set()
    for index, relation in enumerate(relations):
        path = f"relations[{index}]"
        if not isinstance(relation, dict):
            errors.append(f"{path} must be an object")
            continue
        relation_id = relation.get("id")
        if not isinstance(relation_id, str) or not relation_id:
            errors.append(f"{path}.id must be a non-empty string")
        elif relation_id in relation_ids:
            errors.append(f"duplicate relation id: {relation_id}")
        else:
            relation_ids.add(relation_id)
        for field in ("source", "target"):
            if relation.get(field) not in node_by_id:
                errors.append(f"{path}.{field} references missing node: {relation.get(field)!r}")
        if relation.get("source") == relation.get("target"):
            errors.append(f"{path} must not relate a node to itself")
        if relation.get("type") not in RELATION_TYPES:
            errors.append(f"{path}.type is invalid: {relation.get('type')!r}")
        _check_confidence(relation.get("confidence"), f"{path}.confidence", errors)

    uncertainties = graph.get("uncertainties")
    if not isinstance(uncertainties, list):
        errors.append("uncertainties must be an array")
    else:
        for index, uncertainty in enumerate(uncertainties):
            if not isinstance(uncertainty, dict):
                errors.append(f"uncertainties[{index}] must be an object")
                continue
            scope_id = uncertainty.get("scope_id")
            if scope_id is not None and scope_id not in node_by_id:
                errors.append(f"uncertainties[{index}].scope_id references missing node: {scope_id!r}")

    return errors
