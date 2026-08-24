# multicam-bench

A multicamera video pipeline that runs unattended, plus the harness that measures how
many cameras fit on a given machine. See `PROJECT-CHARTER-v2.md` for scope and
rationale.

**Status: v0.1 measured, v0.2–v0.6 code landed but unmeasured** — v0.1: one RTSP
stream, one reader, `ingest_lag` measured from an embedded frame-index marker.
v0.2 adds the N-stream sweep orchestrator (`multicam-bench sweep`); v0.3 adds
`multicam-bench analyze` → `RESULTS.md`; v0.4 adds decode backend
(ffmpeg-cpu/d3d11va/qsv/cuda) and codec (H.264/H.265) as swept axes, with
unavailable backends auto-skipped and recorded; v0.5 adds an optional,
default-off RT-DETRv2 detection stage with its own detect-fps axis and a
golden-configuration accuracy metric; v0.6 adds the two-term capacity fit
(`cost = a·pixel_rate + b·N`) and `multicam-bench calc`. None of v0.2–v0.6 has
been run — no sweep numbers, fitted coefficients, or machine profile exist in
this repo until someone runs `sweep`, `analyze`, and a fit step in that order.
See `PROJECT-CHARTER-v2.md` §7 for the milestone ladder.

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

### v0.4 backend / codec axes

`configs/sweep.yaml`'s `codecs` and `backends` lists are swept as a full cross
product with `n_values`, randomised together per repetition. Requested backends
not available on this machine are auto-detected and skipped before the sweep
starts; the full detection result (including *why* a backend was skipped) is
written to `runs/<id>/backends.json`. On this repo's dev machine (Intel Iris Xe +
NVIDIA MX550, `opencv-python-headless`), expect `ffmpeg-cpu`/`d3d11va`/`qsv`
available and `cuda` skipped — that wheel ships with no CUDA codec support, so a
physically-present NVIDIA GPU does not make NVDEC usable here (see
`rig/backends.py`).

### v0.5 optional detection stage

Off by default (`configs/detect.yaml` `enabled: false`). Turning it on needs the
`detect` extra (`uv sync --extra detect`, installs torch + transformers — not part
of the base install). Detect fps is swept independently of decode fps
(1/2/5/10/25); accuracy is reported as **retention of the most expensive
configuration's vehicle count** on a fixed counting line (`pipeline/counting.py`),
not mAP. The default checkpoint (`PekingU/rtdetr_v2_r18vd`) may not fit this
machine's 2GB VRAM alongside a live decode workload — that's a known limitation,
not a claim it runs well here.

### v0.6 capacity model / calculator

```powershell
uv run multicam-bench calc --cameras 8 --resolution 720p --fps 5 --machine <profile>
```

`<profile>` is a name resolved under `configs/machines/<name>.json` or a direct
path — no profile ships with this repo, since one is only meaningful once built
from real sweep data (`model/fit.py` fits `cost = a·pixel_rate + b·N` per
subsystem with confidence intervals, residuals, and degrees of freedom explicitly
reported; nothing here assumes `b` is zero). `calc` reports fit/no-fit, the
limiting subsystem, and per-subsystem headroom; `fps` is always a required input,
never inferred. Cross-machine prediction (`model/machine_profile.py::predict_cross_machine`)
is refused, not made, when the source and target machines don't share the same
binding subsystem (THREATS-TO-VALIDITY.md T7).
