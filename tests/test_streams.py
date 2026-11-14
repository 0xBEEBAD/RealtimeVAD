"""Quiet streams: more quiet detectors on one VAD (docs/quiet-streams.md)."""

import math
import time

import numpy as np
import pytest

from RealtimeVAD import VAD, Block, Quiet, QuietStream

FRAME = 512
F = FRAME / 16000
B = 0.01


def level(n, amp):
    x = np.full(n, amp, dtype=np.float32)
    x[1::2] *= -1
    return x


def scene(segments):
    """(frames, p, amplitude) -> (scripted probabilities, constant-level audio)."""
    probs, audio = [], []
    for n, p, amp in segments:
        probs += [p] * n
        audio.append(level(n * FRAME, amp))
    return probs, np.concatenate(audio)


def scripted(probs):
    it = iter(probs)
    return lambda frame: next(it, 0.0)


def of(events, stream):
    return [(e.kind, e.time, e.quiet) for e in events if e.stream == stream]


# A word, its ending fading 3.5 dB above the quiet threshold (0.002) while Silero still holds
# it, silence, the next word. Frames 0-9, 10-19, 20-29, 30-39.
WORDS = [(10, 0.95, 0.1), (10, 0.6, 0.003), (10, 0.6, 0.0), (10, 0.95, 0.1)]


def test_streams_with_different_offsets_on_the_same_audio():
    probs, audio = scene(WORDS)
    vad = VAD(probability=scripted(probs))
    vad.add_stream("strict", offset_db=-6.0)
    vad.add_stream("same")
    vad.add_stream("loose", offset_db=6.0)
    ev = vad.feed(audio)
    for name, start in (("strict", 20 * F), ("same", 20 * F), ("loose", 10 * F)):
        (k1, t1, q1), (k2, t2, q2) = of(ev, name)
        assert (k1, k2) == ("quiet", "sound")
        assert t1 == pytest.approx(start + B) and q1.start == pytest.approx(start)
        assert q1.duration == pytest.approx(B) and q1.end is None
        assert t2 == pytest.approx(30 * F + B)  # the first block of the next word
        assert q2.start == pytest.approx(start) and q2.end == pytest.approx(30 * F)
        assert q2.duration == pytest.approx(30 * F - start)
    # the stream without an offset saw the quiet where "pause" did
    pause = next(e for e in ev if e.kind == "pause" and e.source == "quiet")
    assert of(ev, "same")[0][1] == pause.time


def test_quiet_s_holds_the_report_back_and_a_shorter_quiet_reports_nothing():
    probs, audio = scene(WORDS)
    vad = VAD(probability=scripted(probs))
    vad.add_stream("cut", quiet_s=0.15)  # the silence lasts 0.32 s
    vad.add_stream("long", quiet_s=0.5)
    ev = vad.feed(audio)
    (k1, t1, q1), (k2, t2, q2) = of(ev, "cut")
    assert (k1, k2) == ("quiet", "sound")
    assert t1 == pytest.approx(20 * F + 0.15) and q1 == Quiet(pytest.approx(20 * F), pytest.approx(0.15))
    assert q2.duration == pytest.approx(10 * F)
    assert of(ev, "long") == []  # neither "quiet" nor a "sound" without one


def test_rearm_s_keeps_a_click_inside_the_quiet():
    probs, audio = scene([(10, 0.95, 0.1), (20, 0.6, 0.0), (10, 0.95, 0.1)])
    click = 15 * FRAME  # one 10 ms block of sound in the silence
    audio[click : click + 160] = level(160, 0.05)
    vad = VAD(probability=scripted(probs))
    vad.add_stream("any")
    vad.add_stream("held", rearm_s=0.03)
    ev = vad.feed(audio)
    any_, held = of(ev, "any"), of(ev, "held")
    assert [k for k, _, _ in any_] == ["quiet", "sound", "quiet", "sound"]
    assert any_[1][2].end == pytest.approx(15 * F) and any_[2][2].start == pytest.approx(15 * F + B)
    assert [k for k, _, _ in held] == ["quiet", "sound"]
    assert held[1][2].start == pytest.approx(10 * F) and held[1][2].end == pytest.approx(30 * F)
    assert held[1][1] == pytest.approx(30 * F + 0.03)  # decided after rearm_s of sound


def test_settings_change_while_running():
    probs, audio = scene(WORDS)
    vad = VAD(probability=scripted(probs))
    cut = vad.add_stream("cut", quiet_s=0.2)
    half = 15 * FRAME  # in the middle of the fading ending, 3.5 dB over the threshold
    assert vad.feed(audio[:half]) != [] and of(vad.feed(audio[:0]), "cut") == [] and cut.quiet is None
    cut.offset_db = 6.0  # from the next block on the ending counts as quiet
    ev = vad.feed(audio[half : half + 800])
    assert of(ev, "cut") == [] and cut.quiet == Quiet(pytest.approx(15 * F), pytest.approx(0.05))
    cut.quiet_s = 0.08
    ev = vad.feed(audio[half + 800 :])
    (k1, t1, q1), (k2, _, q2) = of(ev, "cut")
    assert k1 == "quiet" and t1 == pytest.approx(15 * F + 0.08) and q1.start == pytest.approx(15 * F)
    assert k2 == "sound" and q2.duration == pytest.approx(15 * F)
    assert cut.quiet is None


def test_streams_run_outside_speech_and_reset_clears_them():
    probs, audio = scene([(10, 0.0, 0.0), (10, 0.95, 0.1), (20, 0.0, 0.0)])
    vad = VAD(probability=scripted(probs + probs))
    room = vad.add_stream("room", quiet_s=0.05)
    ev = vad.feed(audio)
    kinds = [(e.kind, e.stream) for e in ev if e.kind != "pause"]
    assert kinds == [("quiet", "room"), ("sound", "room"), ("start", ""), ("quiet", "room"), ("end", "")]  # "start" needs 64 ms
    assert not vad.in_speech and room.quiet.start == pytest.approx(20 * F)
    vad.reset()
    assert room.quiet is None and vad.streams["room"] is room  # the stream stays, its count is gone
    assert [(e.kind, e.stream) for e in vad.feed(audio) if e.kind != "pause"] == kinds


def test_names_and_the_stream_list():
    vad = VAD(probability=lambda frame: 0.0)
    cut = vad.add_stream("cut", offset_db=3.0)
    assert isinstance(cut, QuietStream) and dict(vad.streams) == {"cut": cut}
    with pytest.raises(ValueError, match="already"):
        vad.add_stream("cut")
    with pytest.raises(ValueError):
        vad.add_stream("")
    with pytest.raises(TypeError):
        vad.streams["other"] = cut
    vad.remove_stream("cut")
    assert not vad.streams and vad.feed(np.zeros(1600, dtype=np.float32)) == []
    with pytest.raises(KeyError):
        vad.remove_stream("cut")


def test_block_events_give_the_level_against_the_threshold():
    probs, audio = scene(WORDS)
    vad = VAD(probability=scripted(probs), block_events=True)
    vad.add_stream("same")
    ev = vad.feed(audio)
    blocks = [e for e in ev if e.kind == "block"]
    assert len(blocks) == len(audio) // 160 and isinstance(blocks[0].block, Block)
    assert [e.time for e in blocks] == pytest.approx([(i + 1) * B for i in range(len(blocks))])
    word, ending, silence = blocks[10].block, blocks[40].block, blocks[70].block
    assert word.level_db == pytest.approx(-20.0, abs=0.1) and word.quiet_db == pytest.approx(-54.0, abs=0.1)
    assert word.over_db == pytest.approx(34.0, abs=0.1)
    assert ending.over_db == pytest.approx(20 * math.log10(0.003 / 0.002), abs=0.1)
    assert silence.over_db < -40
    # a stream is over_db <= offset_db: the same answer block by block
    quiet = [e.time for e in blocks if e.block.over_db <= 0.0]
    (_, t1, q1), (_, _, q2) = of(ev, "same")
    assert quiet[0] == t1 and len(quiet) == round(q2.duration / B)


def test_quiet_clock_off_means_no_stream_ever_reports():
    probs, audio = scene(WORDS)
    vad = VAD(probability=scripted(probs), quiet_rms=0.0, block_events=True)
    vad.add_stream("loose", offset_db=40.0)
    ev = vad.feed(audio)
    assert not [e for e in ev if e.stream] and all(e.block.over_db == math.inf for e in ev if e.kind == "block")


def test_stream_follows_the_noise_subtraction():
    # Steady noise 14 dB under the speaker: no level is under the threshold, the audio is
    # quiet only after subtracting the room. "pause" sees that and so does a stream.
    frames = [40, 30, 40]
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(sum(frames) * FRAME).astype(np.float32) * 10 ** (-54 / 20)
    voice = 0.01 * math.sqrt(2) * np.sin(2 * np.pi * 1000 * np.arange(len(noise)) / 16000)
    voice[: 40 * FRAME] = 0.0
    voice[70 * FRAME :] = 0.0
    probs = [0.1] * 40 + [0.95] * 30 + [0.6] * 12 + [0.1] * 28
    vad = VAD(probability=scripted(probs))
    vad.add_stream("same", quiet_s=0.05)
    ev = vad.feed(noise + voice.astype(np.float32))
    pause = next(e for e in ev if e.kind == "pause")
    assert pause.source == "quiet" and pause.time < 70 * F + 0.05
    after = [q for k, t, q in of(ev, "same") if k == "quiet" and t > 70 * F]
    assert after and after[0].start == pytest.approx(pause.time - B)


# -- the existing events are the same with and without streams -----------------
def plain(events):
    """Every event that exists without streams, with all it carries."""
    out = []
    for e in events:
        if e.kind in ("quiet", "sound", "block"):
            continue
        s, f = e.segment, e.frame
        out.append((e.kind, e.time, e.source,
                    None if s is None else (s.start, s.end, s.speech_end, s.voiced_s, s.peak, s.early, s.audio.tobytes()),
                    None if f is None else (f.probability, f.voiced, f.in_speech, f.silent_frames, f.streak, f.voiced_s, f.peak)))
    return out


def random_scene(seed):
    """About 20 s: words at changing levels with fading endings, clicks, pauses and room noise."""
    rng = np.random.default_rng(seed)
    segments = []
    for _ in range(12):
        amp = float(rng.choice([0.2, 0.05, 0.01, 0.003]))
        segments.append((int(rng.integers(3, 30)), float(rng.choice([0.95, 0.9, 0.6])), amp))
        segments.append((int(rng.integers(1, 6)), 0.6, amp * float(rng.choice([0.02, 0.05, 0.2]))))
        segments.append((int(rng.integers(1, 25)), float(rng.choice([0.6, 0.3, 0.05])), float(rng.choice([0.0, 0.0005, 0.004]))))
    probs, audio = scene(segments)
    noise_db = float(rng.choice([-90.0, -62.0, -48.0]))
    audio = audio + rng.standard_normal(len(audio)).astype(np.float32) * 10 ** (noise_db / 20)
    for at in rng.integers(0, len(audio) - 160, 20):  # 10 ms clicks
        audio[at : at + 160] += level(160, 0.05)
    return probs, audio.astype(np.float32)


@pytest.mark.parametrize("options", [
    {}, {"pause_mode": "fast"}, {"pause_mode": "balanced", "early_end": True}, {"early_end": True, "noise_subtract_db": None},
])
def test_existing_events_are_identical_with_streams(options):
    compared, reported = 0, set()
    for seed in range(12):
        probs, audio = random_scene(seed)
        runs = []
        for streams in (False, True):
            vad = VAD(probability=scripted(probs), resume_events=True, onset_events=True, frame_events=True,
                      block_events=streams, **options)
            if streams:
                vad.add_stream("strict", offset_db=-6.0, quiet_s=0.1)
                vad.add_stream("same")
                vad.add_stream("loose", offset_db=6.0, rearm_s=0.03)
                vad.add_stream("cut", offset_db=12.0, quiet_s=0.3)
            events = []
            for i in range(0, len(audio), 1000):  # chunks that fall on neither a block nor a frame
                events += vad.feed(audio[i : i + 1000])
                if streams and i == 150000:
                    vad.streams["cut"].offset_db = 3.0
            runs.append(events)
        without, with_streams = runs
        assert not [e for e in without if e.kind in ("quiet", "sound", "block")]
        reported |= {e.stream for e in with_streams if e.kind == "quiet"}
        assert plain(without) == plain(with_streams)
        compared += sum(e.kind != "frame" for e in without)
    assert reported == {"strict", "same", "loose", "cut"}
    assert compared > 300  # starts, pauses, resumes, onsets and ends, not only frames


def test_no_stream_costs_nothing_and_four_cost_little():
    # Without Silero (a constant probability), so the detector's own work is what is timed.
    rng = np.random.default_rng(1)
    seconds = 60
    audio = (rng.standard_normal(seconds * 16000) * 0.01).astype(np.float32)
    audio[::2] *= 0.05

    def cost(streams):
        best = math.inf
        for _ in range(3):
            vad = VAD(probability=lambda frame: 0.6)
            for i in range(streams):
                vad.add_stream(f"s{i}", offset_db=3.0 * i, quiet_s=0.01 * (i + 1))
            t0 = time.perf_counter()
            for i in range(0, len(audio), 512):
                vad.feed(audio[i : i + 512])
            best = min(best, time.perf_counter() - t0)
        return best * 1000 / seconds  # ms per second of audio

    none, four = cost(0), cost(4)
    print(f"\nms per second of audio without Silero: no stream {none:.2f}, one {cost(1):.2f}, four {four:.2f}")
    assert four - none < 2.0  # measured: about 0.2 ms
