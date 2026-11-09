import tempfile
import unittest
from pathlib import Path

from research_cli import (
    FULL_TASKS,
    load_prompt_catalog,
    parse_json_candidate,
    safe_model_label,
    sha256_file,
    validate_musicaog_candidate,
)


class ResearchToolTests(unittest.TestCase):
    def test_prompt_catalog_covers_full_suite(self):
        catalog = load_prompt_catalog()
        self.assertEqual(set(FULL_TASKS), set(catalog))
        self.assertTrue(all(item["prompt"].strip() for item in catalog.values()))

    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bin"
            path.write_bytes(b"MOSS-Music")
            self.assertEqual(
                sha256_file(path),
                "40cdcfae7be03e241a01b6220650af735f5f9b77e9bcc6063f2d85502ed2be4c",
            )

    def test_safe_model_label(self):
        self.assertEqual(
            safe_model_label("OpenMOSS-Team/MOSS-Music-8B-Instruct"),
            "OpenMOSS-Team_MOSS-Music-8B-Instruct",
        )

    def test_musicaog_candidate_parser_and_validator(self):
        payload = {
            "schema_version": "0.1",
            "representation": "music_aog_audio_candidate",
            "root": "section-0",
            "nodes": [
                {
                    "id": "section-0",
                    "onset_seconds": 0,
                    "offset_seconds": 10,
                }
            ],
            "productions": [],
            "relations": [],
            "uncertainties": [],
        }
        wrapped = "```json\n" + __import__("json").dumps(payload) + "\n```"
        parsed = parse_json_candidate(wrapped)
        self.assertEqual(parsed["root"], "section-0")
        self.assertEqual(validate_musicaog_candidate(parsed), [])


if __name__ == "__main__":
    unittest.main()
