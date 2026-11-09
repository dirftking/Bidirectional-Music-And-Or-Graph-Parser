"""Bidirectional Music And-Or Graph Parser research utilities.

The package intentionally keeps the audio-language model separate from the
symbolic graph representation: MOSS-Music proposes observations, while this
package parses, assembles, and validates candidate MusicAOG graphs.
"""

PROJECT_NAME = "Bidirectional Music And-Or Graph Parser"
ALGORITHM_NAME = PROJECT_NAME

from .assembler import assemble_candidate_graph
from .audit import audit_task_results
from .groundtruth import load_pop909cl_ground_truth, parse_structure_code, render_pop909cl_score
from .fusion import fuse_moss_mir
from .inference import chord_transition_compatibility, key_chord_compatibility, run_bidirectional_inference
from .metrics import evaluate_artifact, evaluate_graph_artifact, evaluate_mir_artifact, key_accuracy
from .mir import analyze_mir, beat_synchronous_boundaries, estimate_chord_candidates, estimate_key_candidates
from .batch import (
    aggregate_bidirectional_metrics,
    aggregate_metrics,
    aggregate_mir_metrics,
    run_pop909_batch,
    run_pop909_bidirectional_batch,
    run_pop909_mir_batch,
)
from .parsing import JsonExtractionError, extract_json_object
from .validation import validate_graph

__all__ = [
    "PROJECT_NAME",
    "ALGORITHM_NAME",
    "JsonExtractionError",
    "audit_task_results",
    "load_pop909cl_ground_truth",
    "parse_structure_code",
    "render_pop909cl_score",
    "evaluate_artifact",
    "evaluate_graph_artifact",
    "evaluate_mir_artifact",
    "fuse_moss_mir",
    "key_chord_compatibility",
    "chord_transition_compatibility",
    "run_bidirectional_inference",
    "key_accuracy",
    "aggregate_metrics",
    "aggregate_bidirectional_metrics",
    "aggregate_mir_metrics",
    "analyze_mir",
    "beat_synchronous_boundaries",
    "estimate_chord_candidates",
    "estimate_key_candidates",
    "run_pop909_batch",
    "run_pop909_bidirectional_batch",
    "run_pop909_mir_batch",
    "assemble_candidate_graph",
    "extract_json_object",
    "validate_graph",
]
