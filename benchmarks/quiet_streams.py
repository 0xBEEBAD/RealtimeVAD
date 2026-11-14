"""Quiet streams on real audio: are the existing events the same, and what do streams cost?

    python benchmarks/quiet_streams.py a.wav b.wav ... [--seconds 180]

Each 16 kHz mono WAV (its first --seconds) runs through the bundled Silero with no stream,
with one and with four (-6, 0, +6 and +12 dB). Prints, per file and in total: whether every
event that exists without streams (onset, start, pause, resume, end, each frame's
probability, each segment's audio) is identical with four streams and block events, how
many quiets each stream reported, and the CPU time per second of audio (best of --repeat).
"""

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from RealtimeVAD import VAD  # noqa: E402

STREAMS = [("strict", -6.0), ("same", 0.0), ("loose", 6.0), ("cut", 12.0)]


def read(path, seconds):
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return np.frombuffer(w.readframes(int(seconds * 16000)), dtype=np.int16)


def run(audio, streams, block_events=False):
    vad = VAD(resume_events=True, onset_events=True, frame_events=True, block_events=block_events)
    for name, offset in STREAMS[:streams]:
        vad.add_stream(name, offset_db=offset)
    events, t0 = [], time.perf_counter()
    for i in range(0, len(audio), 512):
        events += vad.feed(audio[i : i + 512])
    return events, time.perf_counter() - t0


def plain(events):
    out = []
    for e in events:
        if e.kind in ("quiet", "sound", "block"):
            continue
        s, f = e.segment, e.frame
        out.append((e.kind, e.time, e.source,
                    None if s is None else (s.start, s.end, s.speech_end, s.voiced_s, s.peak, s.early, s.audio.tobytes()),
                    None if f is None else (f.probability, f.voiced, f.in_speech, f.silent_frames, f.streak, f.voiced_s, f.peak)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wavs", nargs="+")
    parser.add_argument("--seconds", type=float, default=180.0)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()
    total_s, cost, same, compared = 0.0, {0: 0.0, 1: 0.0, 4: 0.0}, True, 0
    for path in args.wavs:
        audio = read(path, args.seconds)
        seconds = len(audio) / 16000
        total_s += seconds
        took = {}
        for n in cost:
            runs = [run(audio, n) for _ in range(args.repeat)]
            took[n] = min(t for _, t in runs)
            cost[n] += took[n]
            if n == 0:
                base = runs[0][0]
        full, _ = run(audio, 4, block_events=True)
        equal = plain(base) == plain(full)
        same = same and equal
        kinds = [e.kind for e in base if e.kind != "frame"]
        compared += len(kinds)
        quiets = {name: sum(e.kind == "quiet" and e.stream == name for e in full) for name, _ in STREAMS}
        print(f"{Path(path).name}: {seconds:.0f} s, {kinds.count('start')} segments, {kinds.count('pause')} pauses, "
              f"existing events identical: {equal}; quiets {quiets}; "
              + ", ".join(f"{n} streams {took[n] * 1000 / seconds:.2f}" for n in took) + " ms/s")
    print(f"total: {total_s / 60:.1f} min, {compared} events besides frames, identical: {same}; "
          + ", ".join(f"{n} streams {cost[n] * 1000 / total_s:.2f}" for n in cost) + " ms per second of audio")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
