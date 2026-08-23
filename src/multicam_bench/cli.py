"""CLI entry point: `gen` renders the marked test video, `measure` runs one rig
(mediamtx + publisher + reader) and writes runs/<id>/samples.csv and env.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer

from multicam_bench.bench.env import collect_env
from multicam_bench.bench.reader import measure_stream
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
) -> None:
    """Generate the marked test video used as the publisher's source."""
    path = generate_test_video(output, width, height, fps, duration)
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
