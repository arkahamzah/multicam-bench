"""CLI entry point: `gen` renders the marked test video, `measure` runs one rig
(mediamtx + publisher + reader) and writes runs/<id>/samples.csv and env.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer

from multicam_bench.bench.analyze import run_analyze
from multicam_bench.bench.env import collect_env
from multicam_bench.bench.reader import measure_stream
from multicam_bench.bench.sweep import run_sweep
from multicam_bench.config import load_thresholds
from multicam_bench.rig.generate import generate_test_video
from multicam_bench.rig.probe import probe_video
from multicam_bench.rig.publisher import mediamtx_server, publisher

app = typer.Typer(add_completion=False)


@app.command()
def gen(
    output: Path = typer.Option(Path("data/test_640x360.mp4"), "--output"),
    width: int = typer.Option(640, "--width"),
    height: int = typer.Option(360, "--height"),
    fps: int = typer.Option(30, "--fps"),
    duration: float = typer.Option(60.0, "--duration"),
    codec: str = typer.Option("libx264", "--codec", help="libx264 or libx265"),
) -> None:
    """Generate the marked test video used as the publisher's source."""
    path = generate_test_video(output, width, height, fps, duration, codec=codec)
    typer.echo(f"wrote {path}")


@app.command()
def measure(
    video: Path = typer.Option(Path("data/test_640x360.mp4"), "--video"),
    mediamtx_config: Path = typer.Option(Path("tools/mediamtx.yml"), "--mediamtx-config"),
    thresholds_path: Path = typer.Option(Path("configs/thresholds.yaml"), "--thresholds"),
    rtsp_url: str = typer.Option("rtsp://127.0.0.1:8554/cam0", "--rtsp-url"),
    run_id: str = typer.Option("", "--run-id"),
) -> None:
    """Publish `video` over RTSP and measure ingest_lag / continuity for one reader.

    fps and frame count are read from `video` itself via ffprobe — never passed in —
    so the reader can never disagree with the source about its own pace.
    """
    if not video.exists():
        raise typer.BadParameter(f"{video} does not exist — run `gen` first")

    thresholds = load_thresholds(thresholds_path)
    source = probe_video(video)

    resolved_run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs") / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "env.json").write_text(json.dumps(collect_env(), indent=2), encoding="utf-8")

    with mediamtx_server(mediamtx_config), publisher(video, rtsp_url):
        time.sleep(2.0)  # let the publisher establish the RTSP session before reading
        result = measure_stream(
            rtsp_url=rtsp_url,
            samples_path=run_dir / "samples.csv",
            fps_source=source.fps,
            period_frames=source.frame_count,
            warmup_s=thresholds.warmup_s,
            measure_s=thresholds.measure_s,
        )

    typer.echo(f"wrote {result.samples_path} ({result.frames_measured} samples)")


@app.command()
def sweep(
    sweep_config: Path = typer.Option(Path("configs/sweep.yaml"), "--sweep-config"),
    thresholds_path: Path = typer.Option(Path("configs/thresholds.yaml"), "--thresholds"),
    run_id: str = typer.Option("", "--run-id"),
    seed: int | None = typer.Option(None, "--seed", help="seed for order randomisation"),
) -> None:
    """Run the v0.2 N-stream sweep described by `sweep_config`.

    Long-running: N values × repetitions × (baseline + measured), each on its own
    warmup/measure window plus a cooldown between every run. See
    THREATS-TO-VALIDITY.md for why order is randomised and cooldown enforced.
    """
    sweep_dir = run_sweep(sweep_config, thresholds_path, run_id=run_id, seed=seed)
    typer.echo(f"wrote {sweep_dir}")


@app.command()
def analyze(
    runs_root: Path = typer.Option(Path("runs"), "--runs-root"),
    thresholds_path: Path = typer.Option(Path("configs/thresholds.yaml"), "--thresholds"),
    output: Path = typer.Option(Path("RESULTS.md"), "--output"),
) -> None:
    """Aggregate every sweep run under `runs_root` into `RESULTS.md` (v0.3)."""
    run_analyze(runs_root, thresholds_path, output)
    typer.echo(f"wrote {output}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
