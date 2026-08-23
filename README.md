# multicam-bench

A multicamera video pipeline that runs unattended, plus the harness that measures how
many cameras fit on a given machine. See `PROJECT-CHARTER-v2.md` for scope and
rationale.

**Status: v0.1** — one RTSP stream, one reader, `ingest_lag` measured from an embedded
frame-index marker. See `PROJECT-CHARTER-v2.md` §7 for the milestone ladder.

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
