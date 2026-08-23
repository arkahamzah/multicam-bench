# Prior Art & Positioning

Written before the first release, not after. If this project claimed to be novel,
it would be wrong. Here is what already exists and where this sits.

---

## 1. The research this builds on

A body of systems research has studied the *configuration space* of video analytics
pipelines since 2017. The knobs are the same ones this tool sweeps.

| Work | Venue | What it established |
|---|---|---|
| **VideoStorm** | NSDI '17 | Resource–quality tradeoff over multi-dimensional configurations; offline profiler builds a per-query resource-quality profile; treats *lag* as a first-class goal alongside quality |
| **Chameleon** | SIGCOMM '18 | The optimal configuration (resolution, sampling rate, model) **changes over time** with scene dynamics; exploits temporal and cross-camera correlation to amortise profiling cost |
| **AWStream** | SIGCOMM '18 | Profiles resolution × frame rate × quantisation for bandwidth-adaptive streaming |
| **NoScope / BlazeIt / Focus** | VLDB | Cheap proxy models filter frames before the expensive detector |
| **Reducto / DDS / OneAdapt** | SIGCOMM, arXiv | Cheap heuristics replace expensive periodic profiling |

**Consequence for this project:** the idea of treating resolution and frame rate as
swept parameters is not new, and this README does not claim it is. Every design
choice that echoes these papers is cited rather than presented as original.

## 2. Tools that already exist

| Tool | What it measures | Why it does not answer this question |
|---|---|---|
| `wink-rtsp-bench` | RTSP server capacity — thousands of concurrent connections | Protocol-level. Never decodes a frame |
| `bstreamer`, `rtsp-test-server` | Synthetic RTSP camera emulation | Sources, not consumers |
| `ultravideo/rtp-benchmarks` | RTP library throughput | Transport layer only |
| NVIDIA DeepStream perf tables | Streams per GPU for specific models | Published results, not a tool you can run |
| Frigate NVR | A production NVR | Not instrumented for capacity study |
| MediaMTX + ffmpeg rig | Simulating N cameras | A documented, widely used pattern — used here as infrastructure, not as a contribution |

Existing benchmarks test the **producer** side. None answers the practitioner's
question: *how many cameras can my analytics box actually consume, and what breaks
first.*

## 3. What this project actually contributes

Not an idea. An **artifact**:

1. **A reusable capacity profiler** a practitioner runs on their own machine.
   The research prototypes above profile inside a paper; their harnesses are not
   packaged for reuse.
2. **Decode-path comparison** (CPU / Quick Sync / NVDEC) and **codec comparison**
   (H.264 / H.265) as first-class axes. The literature holds the decode path fixed
   and studies the model; here the decode path is a variable, because on modest
   hardware it is frequently the binding constraint.
3. **Cross-machine capacity model** validated against externally published numbers,
   with predictions published *before* verification.
4. **A capacity calculator** — given a target deployment, does it fit, and what
   saturates first.

## 4. Design consequences of the review

### 4.1 Accuracy is now a measured axis (was missing)

Measuring throughput without accuracy measures nothing useful: any configuration
"fits" if quality is allowed to fall freely. Following the golden-configuration
method used by VideoStorm and Chameleon, the most expensive configuration
(full resolution, full frame rate, strongest detector) serves as pseudo-ground-truth
and cheaper configurations are scored against it. No hand labelling required.

Reported per configuration: capacity **and** the accuracy retained.

### 4.2 Capacity is conditional on scene content

Chameleon showed the optimal frame rate depends on object velocity — stationary
traffic tolerates far lower sampling than free-flowing traffic. Therefore:

- capacity is **not** one number per machine;
- the calculator takes required fps as **input**, and does not attempt to infer it;
- every reported figure names the test content it was measured on.

### 4.3 The linear pixel-rate hypothesis is a hypothesis

`cost = a·(pixel_rate) + b·N`. The per-stream term `b` is measured, never assumed to
be zero. If `b` is large, the model is reported as two-term and the finding is
written up as such.

## 5. Deliberately out of scope

Adaptive configuration at runtime (the actual contribution of Chameleon, AWStream,
Reducto) is **not** implemented. This tool profiles a static configuration space.
Adaptive control is a substantially harder problem and is left cited, not attempted.

---

*Reviewed August 2026. Corrections welcome — if a tool below already does this,
open an issue and this document will be updated.*
