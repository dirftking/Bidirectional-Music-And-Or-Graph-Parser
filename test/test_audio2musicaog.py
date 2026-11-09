import json
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np

from audio2musicaog import (
    assemble_candidate_graph,
    audit_task_results,
    extract_json_object,
    parse_structure_code,
    validate_graph,
)
from audio2musicaog.client import SGLangClient
from audio2musicaog.fusion import fuse_moss_mir
from audio2musicaog.inference import (
    chord_transition_compatibility,
    key_chord_compatibility,
    run_bidirectional_inference,
)
from audio2musicaog.batch import (
    aggregate_bidirectional_metrics,
    aggregate_metrics,
    aggregate_mir_metrics,
    prediction_is_reusable,
)
from audio2musicaog.metrics import (
    boundary_prf,
    chord_weighted_accuracy,
    evaluate_graph_artifact,
    key_accuracy,
    tempo_accuracy,
)
from audio2musicaog.mir import (
    aggregate_chroma_to_beats,
    beat_synchronous_boundaries,
    estimate_chord_candidates,
    estimate_key_candidates,
)
from audio2musicaog.parsing import JsonExtractionError
from audio2musicaog.pipeline import _canonicalize_chord_label, _merge_chord_chunks, retry_failed_tasks


class JsonExtractionTests(unittest.TestCase):
    def test_extracts_final_object_after_thinking_and_example(self):
        text = (
            '<think>An example could be {"wrong": true}.</think>\n'
            '```json\n{"sections": [{"id": "s1"}]}\n```'
        )
        self.assertEqual(extract_json_object(text, {"sections"})["sections"][0]["id"], "s1")

    def test_ignores_braces_inside_json_strings(self):
        text = 'answer: {"sections": [], "note": "literal { brace }"}'
        self.assertEqual(extract_json_object(text, {"sections"})["note"], "literal { brace }")

    def test_missing_required_keys_fails(self):
        with self.assertRaises(JsonExtractionError):
            extract_json_object('{"chords": []}', {"sections"})


class SGLangClientTests(unittest.TestCase):
    def test_native_generate_uses_sampling_seed(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"text": "{}", "meta_info": {}}
        client = SGLangClient("http://localhost:30000")
        client.session.post = Mock(return_value=response)

        client.generate(
            audio_path="fixture.wav",
            prompt="test",
            max_new_tokens=32,
            temperature=0.0,
            seed=7,
        )

        payload = client.session.post.call_args.kwargs["json"]
        self.assertEqual(payload["sampling_params"]["sampling_seed"], 7)
        self.assertNotIn("seed", payload["sampling_params"])

    @patch("audio2musicaog.pipeline.SGLangClient")
    def test_targeted_repair_preserves_original_attempt(self, client_type):
        result = Mock()
        result.text = (
            '{"duration_seconds": 1, "tempo_bpm": 120, "time_signature": "4/4", '
            '"global_key": "C major", "key_changes": [], "confidence": 1, "uncertainties": []}'
        )
        result.meta_info = {"fixture": True}
        client_type.return_value.generate.return_value = result
        with TemporaryDirectory() as temporary:
            from pathlib import Path

            root = Path(temporary)
            audio = root / "audio.wav"
            audio.write_bytes(b"fixture")
            output = root / "prediction.json"
            artifact = {
                "audio": {
                    "path": str(audio), "filename": audio.name,
                    "sha256": "0" * 64, "duration_seconds": 1,
                },
                "server": {"base_url": "http://fixture", "model_info": {"model_path": "fixture"}},
                "generation": {"seed": 42, "temperature": 0, "top_p": 0.95, "top_k": 50},
                "tasks": {
                    "global": {
                        "prompt": "global", "raw_response": "truncated", "meta_info": {},
                        "parsed": None, "parse_error": "fixture parse error",
                    },
                    "structure": {"parsed": {"sections": []}, "parse_error": None},
                    "harmony": {"parsed": {"chords": []}, "parse_error": None},
                },
            }
            repaired = retry_failed_tasks(artifact, output_path=output)

        self.assertEqual(repaired["status"], "complete")
        self.assertIsNone(repaired["tasks"]["global"]["parse_error"])
        attempts = repaired["tasks"]["global"]["attempts"]
        self.assertEqual([item["raw_response"] for item in attempts], ["truncated", result.text])
        self.assertEqual(repaired["repair_runs"][-1]["targeted_tasks"], ["global"])
        self.assertEqual(repaired["repair_runs"][-1]["remaining_parse_errors"], [])


class CrossTaskAuditTests(unittest.TestCase):
    def test_flags_ambiguous_meter_and_key_chord_conflict(self):
        warnings = audit_task_results(
            {"time_signature": "3/4 or 4/4", "global_key": "E minor", "tempo_bpm": 60},
            {"sections": []},
            {
                "chords": [
                    {"start_seconds": 0, "end_seconds": 2, "label": "Ab:maj"},
                    {"start_seconds": 2, "end_seconds": 5, "label": "C:min7"},
                ]
            },
            duration_seconds=5,
        )
        codes = {warning["code"] for warning in warnings}
        self.assertIn("NON_CANONICAL_METER", codes)
        self.assertIn("LOW_KEY_CHORD_CONSISTENCY", codes)

    def test_accepts_simple_diatonic_case(self):
        warnings = audit_task_results(
            {"time_signature": "4/4", "global_key": "C major", "tempo_bpm": 120},
            {"sections": []},
            {
                "chords": [
                    {"start_seconds": 0, "end_seconds": 2, "label": "C:maj"},
                    {"start_seconds": 2, "end_seconds": 4, "label": "G:7"},
                ]
            },
            duration_seconds=4,
        )
        self.assertNotIn("LOW_KEY_CHORD_CONSISTENCY", {warning["code"] for warning in warnings})

    def test_flags_noncanonical_key_and_temporal_coverage(self):
        warnings = audit_task_results(
            {
                "time_signature": "3/4",
                "global_key": "E major or C# minor",
                "tempo_bpm": 60,
                "key_changes": [
                    {"start_seconds": 0, "end_seconds": 0.02, "key": "E major"},
                    {"start_seconds": 0.02, "end_seconds": 0.05, "key": "C# minor"},
                ],
            },
            {"sections": [{"start_seconds": 0, "end_seconds": 5, "label": "intro"}]},
            {
                "chords": [
                    {"start_seconds": 0, "end_seconds": 1, "label": "C#:maj"},
                    {"start_seconds": 1, "end_seconds": 4, "label": "G#:maj"},
                ]
            },
            duration_seconds=5,
        )
        codes = {warning["code"] for warning in warnings}
        self.assertIn("NON_CANONICAL_KEY", codes)
        self.assertIn("KEY_SEGMENT_COVERAGE_GAP", codes)
        self.assertIn("HARMONY_COVERAGE_GAP", codes)


class GroundTruthAndMetricTests(unittest.TestCase):
    def test_parses_compact_human_structure(self):
        result = parse_structure_code("i4A4A2o1", tempo_bpm=120, beats_per_bar=4)
        self.assertEqual(result["total_bars"], 11)
        self.assertEqual([item["label"] for item in result["sections"]], [
            "intro", "section_A", "section_A", "outro"
        ])
        self.assertEqual(result["sections"][0]["end_seconds"], 8.0)
        self.assertEqual(result["sections"][2]["repeated_from"], "gt_section_002")

    def test_rejects_invalid_structure_code(self):
        with self.assertRaises(ValueError):
            parse_structure_code("i4-A4", tempo_bpm=120, beats_per_bar=4)

    def test_tempo_metrics_report_octave_tolerance(self):
        metrics = tempo_accuracy(60, 120)
        self.assertEqual(metrics["acc1"], 0.0)
        self.assertEqual(metrics["acc2"], 1.0)

    def test_key_metric_accepts_enharmonic_spelling(self):
        metrics = key_accuracy("F# major", "Gb major")
        self.assertTrue(metrics["enharmonic_exact"])
        self.assertFalse(metrics["string_exact"])

    def test_chord_metrics_are_duration_weighted(self):
        reference = [
            {"start_seconds": 0, "end_seconds": 2, "label": "C:maj"},
            {"start_seconds": 2, "end_seconds": 4, "label": "G:7"},
        ]
        predicted = [
            {"start_seconds": 0, "end_seconds": 2, "label": "C:maj"},
            {"start_seconds": 2, "end_seconds": 3, "label": "G:min"},
        ]
        metrics = chord_weighted_accuracy(predicted, reference)
        self.assertEqual(metrics["weighted_exact"], 0.5)
        self.assertEqual(metrics["root_accuracy"], 0.75)
        self.assertEqual(metrics["coverage"], 0.75)

    def test_section_boundary_matching_is_one_to_one(self):
        reference = [
            {"start_seconds": 0}, {"start_seconds": 10}, {"start_seconds": 20}
        ]
        predicted = [
            {"start_seconds": 0}, {"start_seconds": 9}, {"start_seconds": 11}, {"start_seconds": 20}
        ]
        metrics = boundary_prf(predicted, reference, tolerance=2)
        self.assertEqual(metrics["matches"], 2)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["precision"], 2 / 3, places=6)

    def test_chunked_harmony_offsets_and_merges_boundary_chord(self):
        merged = _merge_chord_chunks(
            [
                (0, {"chords": [{"start_seconds": 28, "end_seconds": 30, "label": "C:maj", "confidence": 0.9}]}),
                (30, {"chords": [
                    {"start_seconds": "0", "end_seconds": "2.0", "label": "C:maj", "confidence": 0.8},
                    {"start_seconds": "2.0", "end_seconds": "4", "label": "G:7", "confidence": 0.7},
                ]}),
            ]
        )
        self.assertEqual(len(merged["chords"]), 2)
        self.assertEqual(merged["chords"][0]["start_seconds"], 28)
        self.assertEqual(merged["chords"][0]["end_seconds"], 32)
        self.assertEqual(merged["chords"][0]["confidence"], 0.8)
        self.assertEqual(merged["chords"][1]["start_seconds"], 32)
        self.assertEqual(merged["chords"][1]["original_timestamps"]["start_seconds"], "2.0")

    def test_common_chord_spelling_is_canonicalized_without_hiding_source(self):
        self.assertEqual(_canonicalize_chord_label("C#m"), ("C#:min", "C#m"))
        self.assertEqual(_canonicalize_chord_label("F# major"), ("F#:maj", "F# major"))
        self.assertEqual(_canonicalize_chord_label("G:7"), ("G:7", None))

    def test_batch_aggregation_reports_availability(self):
        items = [
            {
                "prediction_status": "complete",
                "metrics": {
                    "prediction_graph_valid": True,
                    "tempo": {"acc1": 1.0, "acc2": 1.0},
                    "time_signature_exact": True,
                    "key_exact": False,
                    "chord": {"weighted_exact": 0.2, "root_accuracy": 0.4, "coverage": 0.8},
                    "section_boundary": {"f1": 0.5},
                    "prediction_audit_warning_count": 2,
                },
            },
            {
                "prediction_status": "partial",
                "metrics": {
                    "prediction_graph_valid": False,
                    "tempo": {"acc1": 0.0, "acc2": 1.0},
                    "time_signature_exact": False,
                    "key_exact": True,
                    "chord": {"weighted_exact": 0.4, "root_accuracy": 0.6, "coverage": 1.0},
                    "section_boundary": {"f1": None},
                    "prediction_audit_warning_count": 4,
                },
            },
            {"status": "failed", "error": "fixture"},
        ]
        summary = aggregate_metrics(items)
        self.assertEqual(summary["songs_requested"], 3)
        self.assertEqual(summary["songs_evaluated"], 2)
        self.assertEqual(summary["prediction_complete_rate"], 0.5)
        self.assertEqual(summary["tempo_acc1"]["mean"], 0.5)
        self.assertEqual(summary["chord_weighted_exact"]["mean"], 0.3)
        self.assertEqual(summary["section_boundary_f1"]["songs_available"], 1)

    def test_running_prediction_is_not_reused(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "prediction.json"
            path.write_text(json.dumps({"status": "running", "tasks": {}}), encoding="utf-8")
            self.assertFalse(prediction_is_reusable(path))
            path.write_text(json.dumps({
                "status": "partial",
                "completed_at_utc": "fixture",
                "tasks": {"global": {}, "structure": {}, "harmony": {}},
                "graph": {},
            }), encoding="utf-8")
            self.assertTrue(prediction_is_reusable(path))

    def test_mir_aggregation_reports_content_metrics(self):
        items = [{"metrics": {
            "tempo": {"acc1": 1.0, "acc2": 1.0},
            "key_exact": True,
            "key_string_exact": False,
            "chord": {"weighted_exact": 0.7, "root_accuracy": 0.8, "coverage": 1.0},
            "section_boundary": {"f1": 0.2},
            "beat_count": 100,
        }}]
        summary = aggregate_mir_metrics(items)
        self.assertEqual(summary["songs_evaluated"], 1)
        self.assertEqual(summary["key_accuracy"]["mean"], 1.0)
        self.assertEqual(summary["key_string_accuracy"]["mean"], 0.0)
        self.assertEqual(summary["chord_weighted_exact"]["mean"], 0.7)

    def test_bidirectional_aggregation_reports_paired_delta(self):
        summary = aggregate_bidirectional_metrics([{
            "before": {"key_exact": False, "chord": {"weighted_exact": 0.4, "root_accuracy": 0.5, "coverage": 1.0}},
            "after": {"key_exact": True, "chord": {"weighted_exact": 0.6, "root_accuracy": 0.7, "coverage": 1.0}},
            "selected_chords_changed": 3,
            "stop_reason": "fixed_point",
        }])
        self.assertEqual(summary["key_accuracy"]["delta"], 1.0)
        self.assertEqual(summary["chord_weighted_exact"]["delta"], 0.2)
        self.assertEqual(summary["selected_chords_changed"], 3)
        self.assertEqual(summary["fixed_point_rate"], 1.0)


class BeatSynchronousMirTests(unittest.TestCase):
    def test_key_profile_recovers_c_major(self):
        chroma = np.repeat(
            np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])[:, None],
            8,
            axis=1,
        )
        self.assertEqual(estimate_key_candidates(chroma)[0]["label"], "C major")

    def test_chord_template_recovers_a_minor(self):
        chroma = np.zeros(12)
        chroma[[9, 0, 4]] = [1.0, 0.8, 0.8]
        candidates = estimate_chord_candidates(chroma)
        self.assertEqual(candidates[0]["label"], "A:min")
        self.assertAlmostEqual(sum(item["confidence"] for item in candidates), 1.0, places=5)

    def test_chroma_is_aggregated_between_detected_beats(self):
        chroma = np.zeros((12, 4))
        chroma[0] = [1, 1, 0, 0]
        chroma[7] = [0, 0, 1, 1]
        intervals = aggregate_chroma_to_beats(
            chroma,
            frame_times=np.asarray([0.0, 0.5, 1.0, 1.5]),
            beat_times=np.asarray([1.0]),
            duration_seconds=2.0,
        )
        self.assertEqual([(item["start_seconds"], item["end_seconds"]) for item in intervals], [(0.0, 1.0), (1.0, 2.0)])
        self.assertEqual(intervals[0]["chroma"][0], 1.0)
        self.assertEqual(intervals[1]["chroma"][7], 1.0)

    def test_structure_boundary_is_on_beat_grid(self):
        intervals = []
        for index in range(40):
            chroma = [0.0] * 12
            chroma[0 if index < 20 else 6] = 1.0
            intervals.append({"start_seconds": float(index), "end_seconds": float(index + 1), "chroma": chroma})
        boundaries = beat_synchronous_boundaries(intervals, max_boundaries=1)
        self.assertEqual(len(boundaries), 1)
        self.assertEqual(boundaries[0]["time_seconds"], 20.0)

    def test_fusion_builds_valid_or_candidates_from_both_sources(self):
        graph = assemble_candidate_graph(
            {"duration_seconds": 2, "global_key": "C major", "confidence": 0.8},
            {"sections": [{
                "id": "s1", "start_seconds": 0, "end_seconds": 2,
                "label": "verse", "confidence": 0.7,
            }]},
            {"chords": [{
                "start_seconds": 0, "end_seconds": 2, "label": "G:7", "confidence": 0.9,
            }]},
            audio_sha256="3" * 64,
            audio_filename="fusion.wav",
            duration_seconds=2,
        )
        moss = {
            "audio": {"sha256": "3" * 64},
            "graph": graph,
            "tasks": {
                "global": {"parsed": {"global_key": "C major", "confidence": 0.8}},
                "harmony": {"parsed": {"chords": [{
                    "start_seconds": 0, "end_seconds": 2, "label": "G:7", "confidence": 0.9,
                }]}},
            },
        }
        mir = {
            "audio": {"sha256": "3" * 64},
            "tempo": {"confidence": 0.9},
            "metrical_grid": {"beats": [
                {"index": 0, "time_seconds": 0.5, "assumed_bar_position": 0},
                {"index": 1, "time_seconds": 1.5, "assumed_bar_position": 1},
            ]},
            "key_candidates": [
                {"label": "C major", "confidence": 0.7},
                {"label": "A minor", "confidence": 0.3},
            ],
            "beat_chord_candidates": [{
                "index": 0, "start_seconds": 0, "end_seconds": 2,
                "candidates": [
                    {"label": "C:maj", "confidence": 0.7},
                    {"label": "A:min", "confidence": 0.3},
                ],
            }],
            "structure_boundary_candidates": [{"time_seconds": 0.5, "confidence": 1.0}],
        }
        fused = fuse_moss_mir(moss, mir)
        self.assertEqual(fused["status"], "complete")
        self.assertEqual(validate_graph(fused["graph"]), [])
        choice = next(node for node in fused["graph"]["nodes"] if node["id"] == "fused_chord_choice_0001")
        productions = [item for item in fused["graph"]["productions"] if item["parent"] == choice["id"]]
        labels_by_id = {node["id"]: node.get("attributes", {}).get("label") for node in fused["graph"]["nodes"]}
        labels = {labels_by_id[item["children"][0]] for item in productions}
        self.assertEqual(labels, {"C:maj", "A:min", "G:7"})
        selected = next(item for item in productions if item["selected"])
        self.assertEqual(labels_by_id[selected["children"][0]], "C:maj")


class BidirectionalInferenceTests(unittest.TestCase):
    def test_key_chord_compatibility_is_interpretable(self):
        self.assertEqual(key_chord_compatibility("C major", "C:maj"), 1.0)
        self.assertAlmostEqual(key_chord_compatibility("C major", "C#:maj"), 1 / 3)
        self.assertEqual(key_chord_compatibility("A minor", "E:7"), 1.0)
        self.assertIsNone(key_chord_compatibility("C major", "not-a-chord"))

    def test_chord_transition_rule_is_explicit(self):
        self.assertEqual(chord_transition_compatibility("C:maj", "C:maj"), 1.0)
        self.assertEqual(chord_transition_compatibility("C:maj", "G:maj"), 0.95)
        self.assertEqual(chord_transition_compatibility("C:maj", "F#:maj"), 0.5)
        self.assertIsNone(chord_transition_compatibility("bad", "C:maj"))

    def test_top_down_key_message_can_change_ambiguous_chord(self):
        graph = assemble_candidate_graph(
            {"duration_seconds": 2, "global_key": "C major", "confidence": 0.8},
            {"sections": [{
                "id": "s1", "start_seconds": 0, "end_seconds": 2,
                "label": "verse", "confidence": 0.7,
            }]},
            {"chords": []},
            audio_sha256="4" * 64,
            audio_filename="inference.wav",
            duration_seconds=2,
        )
        moss = {
            "audio": {"sha256": "4" * 64},
            "graph": graph,
            "tasks": {
                "global": {"parsed": {"global_key": "C major", "confidence": 0.8}},
                "harmony": {"parsed": {"chords": []}},
            },
        }
        mir = {
            "audio": {"sha256": "4" * 64},
            "tempo": {"confidence": 0.9},
            "metrical_grid": {"beats": []},
            "key_candidates": [
                {"label": "C major", "confidence": 0.55},
                {"label": "C# major", "confidence": 0.45},
            ],
            "beat_chord_candidates": [{
                "index": 0, "start_seconds": 0, "end_seconds": 2,
                "candidates": [
                    {"label": "C#:maj", "confidence": 0.55},
                    {"label": "C:maj", "confidence": 0.45},
                ],
            }],
            "structure_boundary_candidates": [],
        }
        fusion = fuse_moss_mir(moss, mir)
        result = run_bidirectional_inference(fusion, key_weight=3.0)

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["stop_reason"], "fixed_point")
        self.assertEqual(result["initial_selection"]["chords"], ["C#:maj"])
        self.assertEqual(result["final_selection"]["chords"], ["C:maj"])
        self.assertEqual(result["selection_changes"]["chords"], 1)
        self.assertEqual(validate_graph(result["graph"]), [])

        ground_truth = {
            "source_id": "fixture",
            "global": {"global_key": "C major", "tempo_bpm": None, "time_signature": None},
            "harmony": {"chords": [{"start_seconds": 0, "end_seconds": 2, "label": "C:maj"}]},
            "structure": {"sections": [{"start_seconds": 0, "end_seconds": 2, "label": "verse"}]},
        }
        metrics = evaluate_graph_artifact(result, ground_truth)
        self.assertTrue(metrics["key_exact"])
        self.assertEqual(metrics["chord"]["weighted_exact"], 1.0)

    def test_sequence_message_decodes_a_global_chord_path(self):
        graph = assemble_candidate_graph(
            {"duration_seconds": 3, "global_key": "C major", "confidence": 1.0},
            {"sections": []},
            {"chords": []},
            audio_sha256="5" * 64,
            audio_filename="sequence.wav",
            duration_seconds=3,
        )
        moss = {
            "audio": {"sha256": "5" * 64},
            "graph": graph,
            "tasks": {
                "global": {"parsed": {"global_key": "C major", "confidence": 1.0}},
                "harmony": {"parsed": {"chords": []}},
            },
        }
        intervals = []
        for index, c_score in enumerate((0.60, 0.49, 0.60)):
            intervals.append({
                "index": index,
                "start_seconds": index,
                "end_seconds": index + 1,
                "candidates": [
                    {"label": "C:maj", "confidence": c_score},
                    {"label": "G:maj", "confidence": 1 - c_score},
                ],
            })
        mir = {
            "audio": {"sha256": "5" * 64},
            "tempo": {"confidence": 1.0},
            "metrical_grid": {"beats": []},
            "key_candidates": [{"label": "C major", "confidence": 1.0}],
            "beat_chord_candidates": intervals,
            "structure_boundary_candidates": [],
        }
        fusion = fuse_moss_mir(moss, mir)
        local = run_bidirectional_inference(fusion, key_weight=0.0, transition_weight=0.0)
        sequence = run_bidirectional_inference(fusion, key_weight=0.0, transition_weight=1.0)

        self.assertEqual(local["final_selection"]["chords"], ["C:maj", "G:maj", "C:maj"])
        self.assertEqual(sequence["artifact_version"], "bidirectional-music-and-or-graph-parser/0.2")
        self.assertEqual(sequence["final_selection"]["chords"], ["C:maj", "C:maj", "C:maj"])
        self.assertEqual(validate_graph(sequence["graph"]), [])


class GraphAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.graph = assemble_candidate_graph(
            {
                "duration_seconds": 20,
                "tempo_bpm": 120,
                "time_signature": "4/4",
                "global_key": "C major",
                "confidence": 0.9,
                "uncertainties": [],
            },
            {
                "sections": [
                    {
                        "id": "s1",
                        "start_seconds": 0,
                        "end_seconds": 10,
                        "label": "intro",
                        "confidence": 0.8,
                        "alternative_labels": [{"label": "verse", "confidence": 0.2}],
                        "repeated_from": None,
                        "evidence": "instrumental opening",
                    },
                    {
                        "id": "s2",
                        "start_seconds": 10,
                        "end_seconds": 20,
                        "label": "verse",
                        "confidence": 0.75,
                        "alternative_labels": [],
                        "repeated_from": "s1",
                        "evidence": "vocal entry",
                    },
                ],
                "uncertainties": [],
            },
            {
                "chords": [
                    {
                        "start_seconds": 0,
                        "end_seconds": 5,
                        "label": "C:maj",
                        "confidence": 0.9,
                        "evidence": "stable C triad",
                    },
                    {
                        "start_seconds": 5,
                        "end_seconds": 10,
                        "label": "G:7",
                        "confidence": 0.7,
                        "evidence": "dominant sonority",
                    },
                ],
                "uncertainties": [],
            },
            audio_sha256="0" * 64,
            audio_filename="fixture.wav",
            duration_seconds=20,
            model="fixture-model",
        )

    def test_assembled_graph_is_valid(self):
        self.assertEqual(validate_graph(self.graph), [])

    def test_ambiguous_label_becomes_or_node(self):
        or_nodes = [node for node in self.graph["nodes"] if node["kind"] == "or"]
        self.assertEqual(len(or_nodes), 2)
        first_choices = [
            item for item in self.graph["productions"] if item["parent"] == or_nodes[0]["id"]
        ]
        self.assertEqual(len(first_choices), 2)
        self.assertAlmostEqual(sum(item["probability"] for item in first_choices), 1.0)
        self.assertEqual(sum(item["selected"] for item in first_choices), 1)

    def test_broken_reference_is_rejected(self):
        broken = json.loads(json.dumps(self.graph))
        broken["relations"][0]["target"] = "missing"
        self.assertTrue(any("missing node" in error for error in validate_graph(broken)))

    def test_probability_sum_is_checked(self):
        broken = json.loads(json.dumps(self.graph))
        parent = next(node["id"] for node in broken["nodes"] if node["kind"] == "or")
        for production in broken["productions"]:
            if production["parent"] == parent:
                production["probability"] = 0.9
        self.assertTrue(any("probabilities must sum" in error for error in validate_graph(broken)))

    def test_zero_duration_section_is_rejected(self):
        broken = json.loads(json.dumps(self.graph))
        section = next(node for node in broken["nodes"] if node["type"] == "section")
        section["span"]["end_seconds"] = section["span"]["start_seconds"]
        self.assertTrue(any("positive duration" in error for error in validate_graph(broken)))

    def test_self_relation_is_rejected(self):
        broken = json.loads(json.dumps(self.graph))
        broken["relations"][0]["target"] = broken["relations"][0]["source"]
        self.assertTrue(any("itself" in error for error in validate_graph(broken)))

    def test_cross_boundary_chord_is_attached_to_song(self):
        graph = assemble_candidate_graph(
            {"duration_seconds": 20, "confidence": 1.0},
            {"sections": [
                {"id": "s1", "start_seconds": 0, "end_seconds": 10, "label": "verse", "confidence": 1.0},
                {"id": "s2", "start_seconds": 10, "end_seconds": 20, "label": "chorus", "confidence": 1.0},
            ]},
            {"chords": [{"start_seconds": 9, "end_seconds": 11, "label": "G:7", "confidence": 1.0}]},
            audio_sha256="1" * 64,
            audio_filename="boundary.wav",
            duration_seconds=20,
        )
        self.assertEqual(validate_graph(graph), [])
        song_production = next(item for item in graph["productions"] if item["parent"] == "song_000")
        self.assertIn("chord_0001", song_production["children"])

    def test_non_positive_observations_are_omitted_with_provenance(self):
        graph = assemble_candidate_graph(
            {"duration_seconds": 20, "confidence": 1.0},
            {"sections": [
                {"id": "bad", "start_seconds": 4, "end_seconds": 4, "label": "verse"},
                {"id": "good", "start_seconds": 5, "end_seconds": 10, "label": "chorus"},
            ]},
            {"chords": [
                {"start_seconds": 2, "end_seconds": 1, "label": "C:maj"},
                {"start_seconds": 10, "end_seconds": 12, "label": "G:7"},
            ]},
            audio_sha256="2" * 64,
            audio_filename="invalid-spans.wav",
            duration_seconds=20,
        )
        self.assertEqual(validate_graph(graph), [])
        self.assertEqual(len([node for node in graph["nodes"] if node["type"] == "section"]), 1)
        self.assertEqual(len([node for node in graph["nodes"] if node["type"] == "chord"]), 1)
        fields = {item["field"] for item in graph["uncertainties"]}
        self.assertIn("structure_observation_omitted", fields)
        self.assertIn("harmony_observation_omitted", fields)


if __name__ == "__main__":
    unittest.main()
