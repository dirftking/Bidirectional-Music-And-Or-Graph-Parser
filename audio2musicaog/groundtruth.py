"""POP909-CL and human-structure adapters for reproducible evaluation."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from bisect import bisect_right
from pathlib import Path
from typing import Any

from .assembler import assemble_candidate_graph


TRIADS = {
    frozenset({0, 4, 7}): "maj",
    frozenset({0, 3, 7}): "min",
    frozenset({0, 3, 6}): "dim",
    frozenset({0, 4, 8}): "aug",
    frozenset({0, 2, 7}): "sus2",
    frozenset({0, 5, 7}): "sus4",
}
SEVENTHS = {
    frozenset({0, 4, 7, 10}): "7",
    frozenset({0, 4, 7, 11}): "maj7",
    frozenset({0, 3, 7, 10}): "min7",
    frozenset({0, 3, 6, 10}): "hdim7",
    frozenset({0, 3, 6, 9}): "dim7",
    frozenset({0, 3, 7, 11}): "minmaj7",
    frozenset({0, 4, 8, 10}): "aug7",
}
PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_key(value: str) -> str:
    value = value.strip()
    if " " in value:
        root, mode, *_ = value.split()
        return f"{root} {'minor' if mode.lower().startswith('min') else 'major'}"
    if value.endswith("m"):
        return f"{value[:-1]} minor"
    return f"{value} major"


def _chord_symbol(notes: list[Any]) -> str | None:
    pitch_classes = {int(note.pitch) % 12 for note in notes}
    if not pitch_classes:
        return "N"
    for root in sorted(pitch_classes):
        intervals = frozenset((pitch - root) % 12 for pitch in pitch_classes)
        quality = SEVENTHS.get(intervals) or TRIADS.get(intervals)
        if quality:
            bass = min(notes, key=lambda note: int(note.pitch)).pitch % 12
            inversion = f"/{PITCH_NAMES[bass]}" if bass != root else ""
            return f"{PITCH_NAMES[root]}:{quality}{inversion}"
    return None


def _select_chord_track(instruments: list[Any]) -> Any:
    candidates = [
        instrument
        for instrument in instruments
        if not instrument.is_drum
        and (int(getattr(instrument, "channel", -1)) == 1 or "chord" in instrument.name.lower())
    ]
    return candidates[-1] if candidates else instruments[-1]


def _tick_converter(tempo_changes: list[Any], ticks_per_beat: int):
    changes = sorted(tempo_changes, key=lambda item: int(item.time))
    if not changes or int(changes[0].time) != 0:
        changes.insert(0, type("Tempo", (), {"time": 0, "tempo": 120.0})())
    ticks = [int(item.time) for item in changes]
    seconds = [0.0]
    for previous, current in zip(changes, changes[1:]):
        delta = int(current.time) - int(previous.time)
        seconds.append(seconds[-1] + delta * 60.0 / (float(previous.tempo) * ticks_per_beat))

    def convert(tick: int) -> float:
        index = max(0, bisect_right(ticks, int(tick)) - 1)
        return seconds[index] + (int(tick) - ticks[index]) * 60.0 / (
            float(changes[index].tempo) * ticks_per_beat
        )

    return convert


def parse_structure_code(
    code: str,
    *,
    tempo_bpm: float,
    beats_per_bar: int,
) -> dict[str, Any]:
    """Parse labels such as ``i4A4B8...o3`` into timed sections."""

    compact = re.sub(r"\s+", "", code)
    tokens = re.findall(r"([A-Za-z])(\d+)", compact)
    if not tokens or "".join(f"{label}{bars}" for label, bars in tokens) != compact:
        raise ValueError(f"invalid POP909 human structure code: {code!r}")
    seconds_per_bar = 60.0 * beats_per_bar / float(tempo_bpm)
    sections: list[dict[str, Any]] = []
    first_by_label: dict[str, str] = {}
    current_bar = 0
    for index, (symbol, bars_text) in enumerate(tokens, start=1):
        bars = int(bars_text)
        section_id = f"gt_section_{index:03d}"
        label = "intro" if symbol == "i" else "outro" if symbol == "o" else f"section_{symbol}"
        sections.append(
            {
                "id": section_id,
                "start_seconds": round(current_bar * seconds_per_bar, 6),
                "end_seconds": round((current_bar + bars) * seconds_per_bar, 6),
                "label": label,
                "symbol": symbol,
                "bars": bars,
                "confidence": 1.0,
                "alternative_labels": [],
                "repeated_from": first_by_label.get(symbol),
                "evidence": "human_label structure annotation",
            }
        )
        first_by_label.setdefault(symbol, section_id)
        current_bar += bars
    return {"sections": sections, "uncertainties": [], "total_bars": current_bar}


def load_pop909cl_ground_truth(
    midi_path: Path,
    *,
    structure_code: str | None = None,
    source_id: str | None = None,
) -> dict[str, Any]:
    """Read corrected POP909-CL metadata/chords and optional human structure."""

    try:
        import miditoolkit
    except ImportError as exc:  # pragma: no cover - exercised on the AutoDL environment
        raise RuntimeError("POP909-CL loading requires: pip install miditoolkit") from exc

    midi_path = midi_path.expanduser().resolve()
    midi = miditoolkit.MidiFile(str(midi_path))
    if not midi.instruments:
        raise ValueError(f"no MIDI instruments found in {midi_path}")
    converter = _tick_converter(list(midi.tempo_changes), int(midi.ticks_per_beat))
    max_tick = max((note.end for instrument in midi.instruments for note in instrument.notes), default=0)
    duration = converter(max_tick)

    chord_track = _select_chord_track(midi.instruments)
    notes_by_start: dict[int, list[Any]] = {}
    for note in chord_track.notes:
        notes_by_start.setdefault(int(note.start), []).append(note)
    chords: list[dict[str, Any]] = []
    for start_tick, notes in sorted(notes_by_start.items()):
        label = _chord_symbol(notes)
        if label is None:
            continue
        end_tick = max(int(note.end) for note in notes)
        chords.append(
            {
                "start_seconds": round(converter(start_tick), 6),
                "end_seconds": round(converter(end_tick), 6),
                "label": label,
                "confidence": 1.0,
                "evidence": "POP909-CL human-corrected chord track",
            }
        )

    tempos = sorted(midi.tempo_changes, key=lambda item: int(item.time))
    signatures = sorted(midi.time_signature_changes, key=lambda item: int(item.time))
    keys = sorted(midi.key_signature_changes, key=lambda item: int(item.time))
    tempo = float(tempos[0].tempo) if tempos else 120.0
    meter = f"{signatures[0].numerator}/{signatures[0].denominator}" if signatures else "4/4"
    key_changes: list[dict[str, Any]] = []
    for index, key in enumerate(keys):
        end_tick = int(keys[index + 1].time) if index + 1 < len(keys) else max_tick
        key_changes.append(
            {
                "start_seconds": round(converter(int(key.time)), 6),
                "end_seconds": round(converter(end_tick), 6),
                "key": _canonical_key(key.key_name),
                "confidence": 1.0,
            }
        )
    global_result = {
        "duration_seconds": round(duration, 6),
        "tempo_bpm": tempo,
        "time_signature": meter,
        "global_key": _canonical_key(keys[0].key_name) if keys else None,
        "key_changes": key_changes,
        "confidence": 1.0,
        "uncertainties": [],
    }
    beats_per_bar = int(signatures[0].numerator) if signatures else 4
    structure_result = (
        parse_structure_code(structure_code, tempo_bpm=tempo, beats_per_bar=beats_per_bar)
        if structure_code
        else {"sections": [], "uncertainties": ["human structure annotation not supplied"]}
    )
    harmony_result = {"chords": chords, "uncertainties": []}
    audio_hash = sha256_file(midi_path)
    graph = assemble_candidate_graph(
        global_result,
        structure_result,
        harmony_result,
        audio_sha256=audio_hash,
        audio_filename=midi_path.name,
        duration_seconds=duration,
        model="POP909-CL-ground-truth",
    )
    return {
        "artifact_version": "audio2musicaog-ground-truth/0.1",
        "dataset": "POP909-CL + POP909 human structure",
        "source_id": source_id or midi_path.stem,
        "source_midi": str(midi_path),
        "midi_sha256": audio_hash,
        "global": global_result,
        "structure": structure_result,
        "harmony": harmony_result,
        "graph": graph,
    }


def render_pop909cl_score(
    midi_path: Path,
    output_wav: Path,
    *,
    score_midi_path: Path | None = None,
    soundfont: Path = Path("/usr/share/sounds/sf2/default-GM.sf2"),
    sample_rate: int = 44100,
) -> dict[str, Any]:
    """Remove the label track before rendering, preventing Ground Truth leakage."""

    try:
        import miditoolkit
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("POP909-CL rendering requires: pip install miditoolkit") from exc
    if not shutil.which("fluidsynth"):
        raise RuntimeError("fluidsynth executable not found")
    midi_path = midi_path.expanduser().resolve()
    output_wav = output_wav.expanduser().resolve()
    score_midi_path = (score_midi_path or output_wav.with_suffix(".score.mid")).expanduser().resolve()
    soundfont = soundfont.expanduser().resolve()
    midi = miditoolkit.MidiFile(str(midi_path))
    if len(midi.instruments) < 2:
        raise ValueError("expected a score track and a chord-label track")
    chord_track = _select_chord_track(midi.instruments)
    midi.instruments = [instrument for instrument in midi.instruments if instrument is not chord_track]
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    score_midi_path.parent.mkdir(parents=True, exist_ok=True)
    midi.dump(str(score_midi_path))
    process = subprocess.run(
        [
            "fluidsynth", "-ni", str(soundfont), str(score_midi_path),
            "-F", str(output_wav), "-r", str(sample_rate), "-g", "0.8",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "source_midi": str(midi_path),
        "score_midi": str(score_midi_path),
        "output_wav": str(output_wav),
        "soundfont": str(soundfont),
        "sample_rate": sample_rate,
        "source_midi_sha256": sha256_file(midi_path),
        "score_midi_sha256": sha256_file(score_midi_path),
        "output_wav_sha256": sha256_file(output_wav),
        "renderer_stderr": process.stderr.strip(),
    }
