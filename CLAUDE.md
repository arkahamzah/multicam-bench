# CLAUDE.md

A multicamera video pipeline that runs unattended, plus the harness that measures how
many cameras fit on a given machine. See `PROJECT-CHARTER-v2.md` for scope and
rationale.

**Everything in this repo is written in English.** Code, comments, commits, docs.

---

## GOVERNING PRINCIPLE

**Anything we do not know becomes a parameter, never an assumption.**

Detection fps, resolution, codec, stream count, target machine — all are sweep axes.
If you find yourself picking a default because "it's probably fine", stop: that value
belongs on an axis.

---

## HARD RULES

1. **No code for anything that cannot be run on this machine.** No DeepStream, no
   physical-camera support, no untested backend abstractions.
2. **Never commit** video files, `runs/`, internal IPs, hostnames, server paths,
   usernames, or org names. `.gitignore` goes in the first commit.
3. **No unbounded queues.** `maxsize` is always set, policy is drop-oldest, drops are
   always counted as a metric.
4. **No `mean` as a headline metric.** p50/p95/p99 or nothing.
5. **Never write "N cameras" without stating resolution and fps.** A stream count
   with no pixel rate attached is meaningless.
6. **Ultralytics is never a default dependency** — AGPL-3.0 is viral, repo is
   Apache-2.0. Optional extras group, documented.
7. **No admin/sudo.** `uv`, portable `mediamtx`, portable `ffmpeg`. If a step needs
   admin, stop and report it.
8. **One process per camera.** Never threads for decode. Failure isolation is a
   product requirement, not an optimisation.
9. **Do not benchmark and refactor in the same session.**

---

## MEASUREMENT CONTRACT

Get these wrong and every number in the repo is worthless.

- **Primary metric is `ingest_lag`** = `t_k − (t₀ + k/fps)`. It is NOT glass-to-glass
  latency. Never label it "latency" unqualified.
- **Discard the first 20 s**, then measure exactly 90 s. This is a 15 W laptop.
  Record start/end temperature. Run on AC power.
- **Every sweep includes a `publisher-only` baseline** at each N; harness cost is
  reported separately.
- **`drop_rate` is only trustworthy with embedded frame indices.** Distinguish "we
  dropped it" from "the decoder skipped it".
- **CUDA timing needs `torch.cuda.synchronize()`** on both sides, or
  `torch.cuda.Event`. Discard 10–20 warm-up iterations.
- **Every run writes `env.json`**: CPU model, core counts, RAM, GPU list, driver
  version, OS build, tool versions. A number without its machine is noise.
- **Fit both terms**: `cost = a·(pixel_rate) + b·N`. Measure `b`; never assume it
  is zero.

---

## STRUCTURE

```
├── README.md              results and plots above the fold
├── METHODOLOGY.md         how measurement works, why naive methods fail
├── RESULTS.md             generated tables, hardware header per machine
├── LICENSES.md
├── src/<pkg>/
│   ├── pipeline/          ingest · detect · track · analytics · sinks
│   ├── ops/               supervisor · watchdog · health/metrics endpoint
│   ├── bench/             rig (mediamtx+ffmpeg) · sweep · analyze
│   └── model/             capacity fit · calculator
├── dashboards/            exported Grafana JSON
├── tests/
├── configs/
└── runs/                  gitignored
```

`ops/` exposes one Prometheus `/metrics` endpoint that serves **both** Grafana and
the benchmark harness. Do not build two metric paths.

---

## WORKING HABITS

- **Stop and ask when a decision is ambiguous.** Do not guess and continue.
- **One milestone at a time.** Do not start v0.3 before v0.2 produces a real CSV from
  a real stream. Do not start the ops layer before v0.4 ships.
- **A real measured number beats a clean abstraction.** When in doubt, get the number.
- Conventional commits, small and focused; the repo works after every one.
- Design rationale goes in `METHODOLOGY.md` — the reasoning is the product.
- Report findings as tables, not paragraphs.
- **When a measurement surprises you, suspect the measurement first.**

---

## BEFORE A LONG RUN

Close Chrome, Teams, and anything else contending for CPU, and record that in
`env.json`. Plug in AC power — battery mode changes the CPU power limit and silently
invalidates comparisons between runs.
