# Quiet streams

*Design note for 0.1.1. The user-facing summary is in the [README](https://github.com/KoljaB/RealtimeVAD/blob/main/README.md#quiet-streams).*

`pause` answers one question with one sensitivity: may the speaker be done?
Other questions about the same audio need another sensitivity, and so far the
only way to get one was a second `VAD`, which runs Silero twice. A quiet
stream is a second (third, fourth …) quiet detector on the same instance. It
reuses what the instance already computes every 10 ms (the block's level, the
room's noise, the speaker's level, the quiet threshold built from them) and
adds only its own comparison and its own count.

## What a stream is

```python
cut = vad.add_stream("cut", offset_db=6.0, quiet_s=0.3)   # returns the stream
cut.offset_db = 9.0                                       # any time, takes effect on the next 10 ms block
for event in vad.feed(chunk):
    if event.kind == "quiet" and event.stream == "cut":   # quiet for quiet_s: event.quiet.start, .duration
        ...
    elif event.kind == "sound" and event.stream == "cut": # that quiet is over: event.quiet.start, .end, .duration
        ...
vad.streams["cut"]; vad.remove_stream("cut")
```

| Setting | Meaning | Default |
|---|---|---|
| `offset_db` | Added to the instance's quiet threshold. `+6` counts audio as quiet 6 dB sooner (more and longer quiets, more of them inside words); negative is stricter than `pause` | `0.0` |
| `quiet_s` | The quiet must have lasted this long before `quiet` is reported | `0.01` (the first block) |
| `rearm_s` | Sound this long ends the quiet. Shorter sound (a click) stays inside it | `0.01` (any sound) |

- **Events.** `quiet` comes once per quiet stretch, when it has lasted
  `quiet_s`. `sound` comes when that stretch is over, and only after a
  `quiet`. Both carry `event.stream` (the name) and `event.quiet`: `start`
  (where the audio went quiet, the beginning of the first quiet block), `end`
  (where the sound came back; `None` on `quiet`) and `duration`. A stretch
  shorter than `quiet_s` reports nothing. `stream.quiet` gives the running
  stretch at any time.
- **Always running.** A stream follows the audio in and out of speech
  segments, because a caller's own unit (a recording, a transcription
  segment) often spans several of them. `vad.in_speech` tells the two apart.
- **Independent.** Streams read the instance's state and write only their
  own. `start`, `end`, `pause`, `resume` and the segment are the same with and
  without them, event by event. Without a stream nothing new runs.
- **Same threshold as `pause`.** A block is quiet for a stream when its level
  is at most `offset_db` over the instance's quiet threshold, or, in steady
  noise close under the speaker, when what the noise subtraction leaves is at
  most `offset_db` over the quiet floor. At `offset_db=0` that is the block
  rule `pause` uses. What a stream does not have is Silero's part of `pause`
  (no `pause` from an unvoiced frame, no "sound in the trailing silence takes
  nothing back").
- **The level itself.** `VAD(block_events=True)` reports every 10 ms block as
  a `block` event with `event.block`: `level_db`, `quiet_db` (the threshold)
  and `over_db` (how far the block is from counting as quiet; at or below 0
  it does). A stream is `over_db <= offset_db` plus a count; a caller who
  wants another rule applies it to `over_db`.

## What several streams are for

One stream per question. A recorder that cuts long turns for transcription
wants cut candidates that get easier to find the longer the open segment is,
while turn detection keeps the strict default: `pause` stays as it is, a
`"cut"` stream runs at `quiet_s=0.3`, and its `offset_db` is raised as the
segment grows. Several fixed streams (0, +3, +6, +9 dB) answer something one
adjustable stream cannot: a stream has no memory of other settings, so after
raising the offset at second 10 it only knows the quiets from then on. With
the tiers running side by side, the caller at second 10 already has every
+6 dB quiet of the last ten seconds and picks the longest. Other uses: a
long, strict stream (`offset_db=-3`, `quiet_s=0.6`) as an input for turn
detection next to the early `pause`; a click-tolerant one (`rearm_s=0.03`)
for measuring pause lengths.

"Any number" holds: a stream costs a comparison and a few integer steps per
10 ms block, so there is no reason to cap them, and names keep the events
apart. In practice two to four are enough; a smooth curve is one stream
whose `offset_db` is set from the caller's own clock, or `over_db`.

## Not included

- No per-stream noise or speaker tracking, and no per-stream `quiet_rms` or
  margins: one offset on the shared threshold is the whole sensitivity.
- No history: a changed setting applies from the next block on.
- No stream ever ends a segment or moves `speech_end`.
