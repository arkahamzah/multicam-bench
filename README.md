# multicam-bench

A multicamera video pipeline that runs unattended, plus the harness that measures how
many cameras fit on a given machine. See `PROJECT-CHARTER-v2.md` for scope and
rationale.

**Status: v0.1 measured, v0.2/v0.3 code landed but unmeasured** — v0.1: one RTSP
stream, one reader, `ingest_lag` measured from an embedded frame-index marker.
v0.2 adds the N-stream sweep orchestrator (`multicam-bench sweep`); v0.3 adds
`multicam-bench analyze` → `RESULTS.md`. Neither has been run yet — no sweep
numbers exist in this repo until someone runs `sweep` then `analyze`. See
`PROJECT-CHARTER-v2.md` §7 for the milestone ladder.

## Setup

Requires `tools/ffmpeg.exe`, `tools/ffprobe.exe`, `tools/mediamtx.exe` on `PATH` for
the session — they are not committed (`.gitignore`), place your own build there.

**PowerShell** (primary — the dev machine for this project is Windows):

```powershell
$env:PATH = "$PWD\tools;$env:PATH"
uv sync
```

**bash** (Linux runners used for the cross-machine study, PROJECT-CHARTER-v2.md §5):

```bash
export PATH="$PWD/tools:$PATH"
uv sync
```

## Running

```powershell
uv run multicam-bench gen                # writes data/test_640x360.mp4 (60s, 640x360, 30fps)
uv run multicam-bench measure            # publishes it over RTSP, reads it back,
                                          # writes runs/<timestamp>/samples.csv + env.json
```

`fps` and frame count for `measure` are read from the video file itself via `ffprobe`
at run time — never passed in and never hardcoded — so the reader cannot disagree
with the source about its own pace.

Before a long run: close Chrome/Teams and anything else contending for CPU, plug in
AC power, and let it run uninterrupted — it discards the configured warm-up window
then measures the configured window, both read from `configs/thresholds.yaml`.

### v0.2 sweep / v0.3 analysis

```powershell
uv run multicam-bench sweep              # N = 1,2,4,8,12,16 (configs/sweep.yaml),
                                          # order randomised per repetition, 3 reps,
                                          # 60s cooldown between runs — long-running
uv run multicam-bench analyze            # reads every runs/<id>/ sweep, writes RESULTS.md
                                          # + fps-vs-N / lag_p95-vs-N plots per sweep
```

`sweep` runs one process per camera (never threads for decode) with a bounded
`maxsize=1` drop-oldest queue between capture and measurement; every N includes a
publisher-only baseline whose CPU cost is recorded separately. Publisher pacing
drift is checked on every measured run against `publisher_drift_reject_ms`
(`configs/thresholds.yaml`) — a run that exceeds it is written `rejected: true` with
a reason and excluded from `analyze`'s tables (THREATS-TO-VALIDITY.md T1).
`analyze` applies the pre-registered saturation criterion (§1.1) to find `N_max`,
excludes frames within ±5 of a loop boundary (T4), and reports median + IQR, never
a mean.
