# Threats to Validity & Experimental Protocol

Written before data collection, not after. Every threshold below is fixed in advance
so that results cannot be tuned to look good once the numbers are in.

---

## 1. Pre-registered definitions

### 1.1 Saturation criterion

A configuration is **saturated** when, over a 30 s sustained window after warm-up,
either holds:

- `ingest_lag` p95 exceeds **200 ms**, or
- `drop_rate` exceeds **1.0 %**

`N_max` is the largest stream count where **no** camera is saturated. Both thresholds
are fixed here and must not be changed after seeing results. If a threshold turns out
to be wrong, the change is recorded with its date and rationale, and **all affected
results are re-run**.

### 1.2 Ingest lag

```
elapsed_frames(t) = unwrap(embedded_index)          # NOT the received-frame count
lag                = t_recv − (t_anchor + elapsed_frames / fps_source)
```

`t_anchor` is set **after** the warm-up window, never on the first frame.

> **Why this matters.** Deriving elapsed time from the *received* frame count
> under-reports lag exactly when frames are being dropped — that is, precisely at
> saturation. The metric would dampen the signal it exists to detect. Elapsed time
> must come from the embedded index, which counts frames the source *sent*.

### 1.3 Accuracy

Golden configuration (max resolution, max fps, strongest detector) serves as
pseudo-ground-truth, following the method used by VideoStorm and Chameleon.

Primary metric: **vehicle count error** on a fixed counting line, not mAP. Counting
is what the application produces; mAP is a proxy for it.

Cross-fps comparison is performed on the **intersecting frame set only**, or on the
downstream count, never frame-by-frame across differing sample rates.

---

## 2. Threats to internal validity

### T1 — Publisher starvation masquerading as consumer saturation (**severe**)

Publishers and consumer share one machine. Under load, `ffmpeg -re` may fail to pace
accurately; the resulting lag growth is indistinguishable from consumer saturation.

**Control.** Publisher pacing accuracy is measured at every N via the embedded index
timeline. If publisher drift exceeds **50 ms** over the measurement window, the data
point is **rejected**, not reported. Rejected points appear in results as gaps with
the reason stated.

This caps the usable N on any single machine. That cap is a property of the method
and is reported, not hidden.

### T2 — Thermal drift and run ordering

A 15 W laptop throttles. Sweeping N in ascending order confounds load with heat.

**Control.** Configuration order randomised per repetition; ≥ 60 s cooldown between
runs; package temperature recorded at start and end of every run; runs on AC power
with the Windows power plan pinned to High Performance.

### T3 — No variance estimate

**Control.** Minimum **3 repetitions** per configuration. Report **median and IQR**,
never a single value. A difference smaller than the observed IQR is reported as
"no detectable difference", not as an improvement.

### T4 — Loop boundary artefacts

The source loops with `-stream_loop -1`. Timestamp and GOP discontinuities at the
loop point can produce lag spikes unrelated to system load.

**Control.** Frames within ±5 of a loop boundary are excluded, and the count of
excluded frames is reported.

### T5 — Harness cost attribution

**Control.** Every sweep includes a publisher-only baseline at each N. mediamtx CPU
is measured as a separate process, not folded into either side.

### T6 — Backend behaviour that cannot be assumed

`CAP_PROP_BUFFERSIZE` is not honoured by every backend, and OpenCV may drop frames
internally.

**Control.** Embedded-index continuity is checked on every run. A run whose
continuity is unexplained by measured drops is discarded as untrustworthy.

---

## 3. Threats to external validity

### T7 — Cross-machine extrapolation is regime-bound (**severe**)

The capacity model is fitted per subsystem. Predicting machine B from machine A is
valid **only when the same subsystem is the binding constraint on both**. A
CPU-bound laptop tells you little about a GPU-bound server.

**Control.** Every prediction states the assumed binding subsystem. Predictions
across a regime change are labelled invalid and are not made.

### T8 — Synthetic content is not traffic

`testsrc2` has encoding and detection characteristics unlike a real CCTV scene.

**Control.** Every result names its test content. All headline numbers are
reproduced on at least one real, openly licensed traffic clip. Where synthetic and
real diverge, both are reported.

### T9 — Capacity is content-dependent

Chameleon demonstrated that the frame rate required for a given accuracy varies with
scene dynamics; stationary traffic tolerates far lower sampling than free-flowing
traffic.

**Control.** The calculator takes required fps as an **input** and never infers it.
No single-number "capacity of machine X" is published without its accuracy target
and test content.

### T10 — Golden-config pseudo-ground-truth inherits detector bias

Errors the strongest detector makes are treated as truth.

**Control.** Stated explicitly. Accuracy figures are reported as *retention relative
to the golden configuration*, never as absolute accuracy.

---

## 4. Threats to construct validity

### T11 — `ingest_lag` is not glass-to-glass latency

It excludes sensor, ISP, encoder, real network, and RTSP server buffering. It
measures how far the consumer has fallen behind the source's pace.

**Control.** Named `ingest_lag` everywhere in code, docs, and plots. The word
"latency" is never used unqualified.

### T12 — nvidia-smi utilisation is not occupancy

Reported GPU "utilisation" is the fraction of time at least one kernel was resident,
not the fraction of the device in use. 100 % can coexist with low SM occupancy.

**Control.** Utilisation is used only as a saturation *indicator*, never as a
capacity measure. NVDEC utilisation is read separately (`nvidia-smi dmon -s um`).

---

## 5. Model specification

```
cost = a · pixel_rate + b · N          pixel_rate = W × H × fps_detect
```

`b` — the per-stream overhead — is **measured, never assumed zero**. Report both
coefficients with confidence intervals and the residuals. If `b` dominates, say so:
the finding that per-stream overhead outweighs pixel throughput would be more
interesting than a clean linear fit.

Degrees of freedom are reported. Fitting two parameters to five noisy points is
stated as such rather than presented as an established law.

---

## 6. What would falsify the central claim

Stated in advance, so the project can be wrong in a way that is visible:

1. If `b` dominates `a`, capacity is governed by stream count, not pixel rate, and
   the core hypothesis is wrong.
2. If Quick Sync does **not** beat NVDEC on this hardware, the headline expectation
   fails and the result is published unchanged.
3. If cross-machine prediction error exceeds 30 % within a single bottleneck regime,
   the model does not generalise and the calculator is withdrawn.

Each of these is reported whether or not it is the flattering outcome.


---

## 7. Known measurement-path limitations (added after v0.4)

### T13 — `cv2.VideoCapture` cannot reach hardware decode on this OpenCV build (**severe**)

`opencv-python-headless` (this repo's pinned wheel) reports 0 CUDA-enabled
devices via `cv2.cuda.getCudaEnabledDeviceCount()` regardless of what GPU is
physically installed — that wheel ships with no CUDA codec support at all. NVDEC
is therefore unreachable through `cv2.VideoCapture`'s `CAP_PROP_HW_ACCELERATION`
on any machine using this wheel, which is the default install for this repo. The
`d3d11va`/`qsv`/`vaapi` acceleration flags cv2 does expose may also silently
fall back to software decode rather than raising, which is difficult to
distinguish from ordinary pipeline overhead by watching CPU usage alone.

This removes the single most valuable axis in PROJECT-CHARTER-v2.md §2 ("backend
decode": cpu / qsv / d3d11va / nvdec) if the RTSP/cv2 pipeline sweep
(`bench/sweep.py`) is the only measurement path — a `cuda` data point measured
through cv2 on this wheel is not measuring NVDEC, it is measuring software decode
mislabelled as hardware decode.

**Control.** A second, structurally separate measurement path
(`rig/ffmpeg_decode_bench.py`, `bench/decode_only_sweep.py`,
`multicam-bench sweep --decode-only`) benchmarks decode directly via
`ffmpeg -hwaccel <backend> -i <file> -benchmark -f null -` as a subprocess, with
no cv2 and no RTSP involved. This is the same acceleration path DeepStream and
most published benchmarks use, so it is also the more externally-comparable
number (§5 above). Its results are reported in a separate `RESULTS.md` table
(`bench/analyze_decode_only.py`) and are never merged cell-for-cell with the
full-pipeline table — they measure different things (raw decode throughput vs.
effective consumer fps / ingest_lag under real RTSP ingest and queueing) and a
merged table would misrepresent both.

**Residual risk.** The decode-only path still depends on ffmpeg itself being
built with working `-hwaccel` support for this machine's driver stack (CUDA
toolkit / Intel Media SDK / VAAPI userspace driver). A backend reported
"available" by `rig/backends.py` (which checks OpenCV/GPU presence, not ffmpeg's
own hwaccel support) can still fail inside `run_ffmpeg_decode_benchmark` — that
failure is recorded per-point (`succeeded: false`, `error`) rather than silently
dropped, but it is not pre-empted by the availability check.
