"""RF-DETR vehicle detection, tracking, and traffic-signal dashboard.
Keep vision_tracking.py in this folder.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

from vision_tracking import build_detector


# ---------------- EASY SETTINGS ----------------

# Positive values advance camera 1's footage.
# At output time 10 seconds, 1.0 reads camera 1 at 11 seconds.
CAMERA_1_ADVANCE_SECONDS = 1.0

# 0 preserves the source frame rate.
# Set to 10, or use --output-fps 10, for fewer detector calls.
OUTPUT_FPS = 0.0

BOX_THICKNESS = 2
LABEL_FONT_SCALE = 0.43

TILE_WIDTH, TILE_HEIGHT = 960, 540

GAP_OUT_SECONDS = 3600 / 1900 * 2
MIN_GREEN_SECONDS = 5.0
INITIAL_GREEN_SECONDS = 10.0
YELLOW_SECONDS = 5.0
MAX_GREEN_SECONDS = 120.0

VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm",
    ".m4v", ".wmv", ".mpeg", ".mpg",
}

BOX_COLOR = (90, 210, 130)

SIGNAL_COLORS = {
    "GREEN": (80, 210, 100),
    "YELLOW": (60, 210, 230),
    "RED": (80, 80, 225),
}


def integer_list(value):
    return [int(item.strip()) for item in value.split(",")]


def find_videos(path, extraction_root):
    source = Path(path)

    if not source.exists():
        raise FileNotFoundError(
            f"Video path does not exist: {source}"
        )

    if source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            root = Path(extraction_root).resolve()

            for member in archive.infolist():
                destination = (root / member.filename).resolve()

                if not destination.is_relative_to(root):
                    raise ValueError(
                        "ZIP contains an invalid relative path"
                    )

            archive.extractall(root)

        source = root

    if source.is_dir():
        videos = sorted(
            item
            for item in source.rglob("*")
            if item.is_file()
            and item.suffix.lower() in VIDEO_EXTENSIONS
        )
    else:
        videos = [source]

    if not videos:
        raise ValueError(f"No videos found under {source}")

    return videos


class VideoReader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))

        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"Could not open video: {path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        if not math.isfinite(self.fps) or self.fps <= 0:
            self.fps = 30.0

        self.index = -1

    def frame_at(self, source_seconds):
        target = int(round(source_seconds * self.fps))

        if target < self.index:
            raise ValueError(
                "Video timestamps must move forward"
            )

        # Skip frames without retrieving every full image.
        while self.index < target:
            if not self.cap.grab():
                return None

            self.index += 1

        ok, frame = self.cap.retrieve()
        return frame if ok else None

    def close(self):
        self.cap.release()


class PhaseController:
    def __init__(self, initial_phase):
        self.phase_index = initial_phase
        self.state = "GREEN"
        self.started = 0.0
        self.target = None
        self.initial = True
        self.last_active = 0.0
        self.last_decision = -1.0

    def update(self, demands, t):
        if self.state == "YELLOW":
            if t - self.started >= YELLOW_SECONDS:
                self.phase_index = self.target
                self.target = None
                self.state = "GREEN"
                self.started = t
                self.last_active = t
                self.initial = False
            else:
                return

        if demands[self.phase_index] > 0:
            self.last_active = t

        if t - self.last_decision < 1.0:
            return

        self.last_decision = t

        minimum = (
            INITIAL_GREEN_SECONDS
            if self.initial
            else MIN_GREEN_SECONDS
        )

        if t - self.started < minimum:
            return

        gapped = (
            t - self.last_active >= GAP_OUT_SECONDS
        )
        maxed = (
            t - self.started >= MAX_GREEN_SECONDS
        )

        if not gapped and not maxed:
            return

        others = [
            i for i in range(4)
            if i != self.phase_index
        ]
        best = max(others, key=lambda i: demands[i])

        if demands[best] > 0:
            self.target = best
            self.state = "YELLOW"
            self.started = t

    def color_for(self, phase):
        if phase == self.phase_index:
            return self.state

        return "RED"


def draw_view(
    frame,
    boxes,
    ids,
    camera_number,
    phase,
    count,
    controller,
    source_time,
    output_time,
    args,
):
    tile = cv2.resize(
        frame, (TILE_WIDTH, TILE_HEIGHT)
    )

    height, width = frame.shape[:2]

    for box, track_id in zip(boxes, ids):
        x1, y1, x2, y2 = box

        x1, x2 = [
            int(np.clip(
                round(x * TILE_WIDTH / width),
                0,
                TILE_WIDTH - 1,
            ))
            for x in (x1, x2)
        ]

        y1, y2 = [
            int(np.clip(
                round(y * TILE_HEIGHT / height),
                0,
                TILE_HEIGHT - 1,
            ))
            for y in (y1, y2)
        ]

        cv2.rectangle(
            tile,
            (x1, y1),
            (x2, y2),
            BOX_COLOR,
            args.box_thickness,
            cv2.LINE_AA,
        )

        label = f"C{camera_number}-{track_id}"

        label_width, label_height = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            args.label_scale,
            1,
        )[0]

        left = min(
            x1,
            max(0, TILE_WIDTH - label_width - 3),
        )

        baseline = (
            y1 - 5
            if y1 > label_height + 5
            else min(
                TILE_HEIGHT - 3,
                y2 + label_height + 5,
            )
        )

        # Small text halo improves visibility.
        cv2.putText(
            tile,
            label,
            (left, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            args.label_scale,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            tile,
            label,
            (left, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            args.label_scale,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

    signal = controller.color_for(phase)
    color = SIGNAL_COLORS[signal]

    cv2.rectangle(
        tile,
        (1, 1),
        (TILE_WIDTH - 2, TILE_HEIGHT - 2),
        color,
        2,
    )

    cv2.rectangle(
        tile,
        (4, 4),
        (TILE_WIDTH - 5, 52),
        (20, 20, 20),
        -1,
    )

    header = (
        f"CAM {camera_number} | PHASE {phase} | "
        f"{signal} | VEHICLES {count}"
    )

    detail = (
        f"Video {output_time:.1f}s | "
        f"Source {source_time:.1f}s | Camera-local IDs"
    )

    cv2.putText(
        tile,
        header,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        color,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        tile,
        detail,
        (12, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    return tile


def run(args):
    start = time.perf_counter()

    with tempfile.TemporaryDirectory(
        prefix="traffic_video_"
    ) as temporary:
        all_videos = find_videos(
            args.video, temporary
        )

        selected = args.include_cameras

        if selected is None:
            selected = list(range(min(
                4 if args.merge_views else 1,
                len(all_videos),
            )))

        if (
            not selected
            or len(set(selected)) != len(selected)
            or any(
                index not in range(len(all_videos))
                for index in selected
            )
        ):
            raise ValueError(
                "--include-cameras needs unique "
                "existing camera indices"
            )

        videos = [
            Path(all_videos[index])
            for index in selected
        ]

        mapping = args.camera_phase_map or [
            i % 4 for i in range(len(videos))
        ]

        if (
            len(mapping) != len(videos)
            or any(
                phase not in range(4)
                for phase in mapping
            )
        ):
            raise ValueError(
                "Provide one phase (0..3) "
                "per selected camera"
            )

        advances = [
            args.camera_1_advance if index == 0 else 0.0
            for index in selected
        ]

        output_dir = Path(args.output_dir)
        output_dir.mkdir(
            parents=True, exist_ok=True
        )

        output_path = (
            Path(args.output)
            if args.output
            else output_dir / "combined_intersection_demo.mp4"
        )

        log_path = output_dir / "camera_counts.csv"
        tracks_path = output_dir / "tracks.csv"

        output_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        if any(
            video.resolve() == output_path.resolve()
            for video in videos
        ):
            raise ValueError(
                "Output video must not overwrite "
                "an input video"
            )

        print(
            "Starting vehicle detection and tracking.",
            flush=True,
        )

        for i, video in enumerate(videos):
            print(
                f"CAM {selected[i] + 1}: {video} | "
                f"source advance +{advances[i]:.1f}s | "
                f"phase {mapping[i]}",
                flush=True,
            )

        readers = []
        writer = None
        frames_written = 0
        inference_calls = 0
        interrupted = False

        try:
            for video in videos:
                readers.append(VideoReader(video))

            source_fps = min(
                reader.fps for reader in readers
            )

            fps = (
                min(args.output_fps, source_fps)
                if args.output_fps > 0
                else source_fps
            )

            print(
                f"Processing and saving {fps:g} "
                "frames per video second.",
                flush=True,
            )

            first = build_detector(
                args.yolo_model,
                args.conf_threshold,
                args.rfdetr_classes,
                args.nms_iou,
                args.track_max_age,
                args.track_center_gate,
                args.device,
            )

            # Share inference resources; keep separate trackers.
            detectors = [first] + [
                first.fork() for _ in videos[1:]
            ]

            columns = (
                2 if len(videos) == 4
                else min(3, len(videos))
            )
            rows = (
                len(videos) + columns - 1
            ) // columns

            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (
                    columns * TILE_WIDTH,
                    rows * TILE_HEIGHT,
                ),
            )

            if not writer.isOpened():
                raise RuntimeError(
                    f"Could not write {output_path}. "
                    "Close any player using it."
                )

            controller = PhaseController(
                args.initial_phase
            )

            with (
                log_path.open("w", newline="") as log,
                tracks_path.open("w", newline="") as tracks,
            ):
                log_writer = csv.writer(log)
                log_writer.writerow([
                    "video_time",
                    "active_phase",
                    "signal_state",
                    "camera_counts",
                    "phase_demands",
                ])

                track_writer = csv.writer(tracks)
                track_writer.writerow([
                    "video_time",
                    "camera",
                    "source_time",
                    "track_id",
                    "x1", "y1", "x2", "y2",
                    "confidence",
                ])

                try:
                    while frames_written / fps < args.duration:
                        t = frames_written / fps

                        frames = [
                            reader.frame_at(t + advance)
                            for reader, advance in zip(
                                readers, advances
                            )
                        ]

                        if any(
                            frame is None
                            for frame in frames
                        ):
                            print(
                                "Reached the end of "
                                "an input video.",
                                flush=True,
                            )
                            break

                        results = []
                        counts = []

                        for frame, detector in zip(
                            frames, detectors
                        ):
                            # Draw current detections only.
                            inference_calls += 1
                            result = detector.track(
                                frame, timestamp=t
                            )[0]

                            boxes = (
                                result.boxes.xyxy
                                .cpu().numpy()
                            )

                            ids = (
                                result.boxes.id
                                .cpu().numpy().astype(int)
                            )

                            confidence = (
                                result.boxes.conf
                                .cpu().numpy()
                            )

                            results.append(
                                (boxes, ids, confidence)
                            )
                            counts.append(len(boxes))

                        demands = [0] * 4

                        for i, count in enumerate(counts):
                            demands[mapping[i]] += count

                        controller.update(demands, t)

                        dashboard = np.zeros(
                            (
                                rows * TILE_HEIGHT,
                                columns * TILE_WIDTH,
                                3,
                            ),
                            np.uint8,
                        )

                        for i, (
                            frame,
                            (boxes, ids, confidence),
                        ) in enumerate(zip(frames, results)):
                            tile = draw_view(
                                frame,
                                boxes,
                                ids,
                                selected[i] + 1,
                                mapping[i],
                                counts[i],
                                controller,
                                t + advances[i],
                                t,
                                args,
                            )

                            row, column = divmod(
                                i, columns
                            )

                            dashboard[
                                row * TILE_HEIGHT:
                                (row + 1) * TILE_HEIGHT,
                                column * TILE_WIDTH:
                                (column + 1) * TILE_WIDTH,
                            ] = tile

                            for box, track_id, score in zip(
                                boxes, ids, confidence
                            ):
                                track_writer.writerow([
                                    round(t, 3),
                                    selected[i] + 1,
                                    round(t + advances[i], 3),
                                    int(track_id),
                                    *[
                                        round(float(value), 2)
                                        for value in box
                                    ],
                                    round(float(score), 4),
                                ])

                        writer.write(dashboard)

                        log_writer.writerow([
                            round(t, 3),
                            controller.phase_index,
                            controller.state,
                            json.dumps(counts),
                            json.dumps(demands),
                        ])

                        frames_written += 1

                        if (
                            frames_written
                            % max(1, round(fps)) == 0
                        ):
                            elapsed = (
                                time.perf_counter() - start
                            )
                            completed = frames_written / fps

                            remaining = (
                                max(0, args.duration - completed)
                                * elapsed
                                / max(completed, 1e-6)
                            )

                            print(
                                f"Video {completed:.1f}/"
                                f"{args.duration:.0f}s | "
                                f"counts={counts} | "
                                f"wall time {elapsed:.0f}s | "
                                "estimated remaining "
                                f"{remaining:.0f}s",
                                flush=True,
                            )

                        if args.show:
                            cv2.imshow(
                                "Traffic dashboard",
                                dashboard,
                            )

                            if cv2.waitKey(1) & 255 == 27:
                                interrupted = True
                                break

                except KeyboardInterrupt:
                    interrupted = True
                    print(
                        "\nStopping; finalizing "
                        "the partial video.",
                        flush=True,
                    )

        finally:
            for reader in readers:
                reader.close()

            if writer is not None:
                writer.release()

            if args.show:
                cv2.destroyAllWindows()

        elapsed = time.perf_counter() - start

        manifest = {
            "model": Path(args.yolo_model).name,
            "inputs": [
                f"{video.parent.name}/{video.name}"
                for video in videos
            ],
            "original_camera_indices": selected,
            "source_advances_seconds": advances,
            "source_fps": [
                reader.fps for reader in readers
            ],
            "output_fps": fps,
            "frames_written": frames_written,
            "video_seconds": frames_written / fps,
            "processing_seconds": round(elapsed, 3),
            "interrupted": interrupted,
            "detector_calls": inference_calls,
            "cross_camera_matching": False,
            "id_scope": "camera-local",
            "count_basis": (
                "current-frame vehicle presence per camera"
            ),
            "controller": "actuated",
            "camera_phase_map": mapping,
            "confidence_threshold": args.conf_threshold,
            "nms_iou": args.nms_iou,
            "output_video": output_path.name,
        }

        (output_dir / "demo_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )

        if frames_written == 0:
            raise RuntimeError(
                "No frames written. Check videos "
                "and camera 1 advance."
            )

        print(
            f"Finished: {output_path}",
            flush=True,
        )
        print(
            f"Counts: {log_path}\nTracks: {tracks_path}",
            flush=True,
        )
        print(
            f"{frames_written / fps:.1f}s of footage "
            f"processed in {elapsed:.1f}s.",
            flush=True,
        )


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__
    )

    p.add_argument("--video", required=True)
    p.add_argument(
        "--yolo-model",
        "--model",
        dest="yolo_model",
        required=True,
    )
    p.add_argument(
        "--controller",
        choices=["actuated"],
        default="actuated",
    )
    p.add_argument(
        "--merge-views", action="store_true"
    )
    p.add_argument(
        "--include-cameras", type=integer_list
    )
    p.add_argument(
        "--camera-phase-map", type=integer_list
    )
    p.add_argument(
        "--camera-1-advance",
        type=float,
        default=CAMERA_1_ADVANCE_SECONDS,
    )
    p.add_argument(
        "--output-fps",
        type=float,
        default=OUTPUT_FPS,
        help=(
            "0 preserves source FPS; "
            "use 10 for fewer detector calls"
        ),
    )
    p.add_argument(
        "--box-thickness",
        type=int,
        default=BOX_THICKNESS,
    )
    p.add_argument(
        "--label-scale",
        type=float,
        default=LABEL_FONT_SCALE,
    )
    p.add_argument(
        "--duration", type=float, default=120
    )
    p.add_argument(
        "--conf-threshold",
        type=float,
        default=0.30,
    )
    p.add_argument(
        "--nms-iou", type=float, default=0.50
    )
    p.add_argument(
        "--track-max-age",
        type=float,
        default=0.5,
    )
    p.add_argument(
        "--track-center-gate",
        type=float,
        default=2.0,
    )
    p.add_argument("--rfdetr-classes")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--initial-phase", type=int, default=0
    )
    p.add_argument(
        "--output-dir",
        default="traffic_demo_output",
    )
    p.add_argument("--output")
    p.add_argument(
        "--show", action="store_true"
    )

    args = p.parse_args()

    values = [
        args.camera_1_advance,
        args.output_fps,
        args.label_scale,
        args.duration,
        args.conf_threshold,
        args.nms_iou,
        args.track_max_age,
        args.track_center_gate,
    ]

    if any(
        not math.isfinite(value)
        for value in values
    ):
        p.error("Numeric settings must be finite")

    if (
        min(
            args.camera_1_advance,
            args.output_fps,
        ) < 0
        or min(
            args.label_scale,
            args.duration,
            args.box_thickness,
            args.track_max_age,
            args.track_center_gate,
        ) <= 0
    ):
        p.error(
            "Advance and output FPS must be nonnegative; "
            "other limits must be positive"
        )

    if not all(
        0 < value <= 1
        for value in (
            args.conf_threshold, args.nms_iou
        )
    ):
        p.error(
            "Confidence and NMS thresholds "
            "must be in (0,1]"
        )

    if args.initial_phase not in range(4):
        p.error("Initial phase must be 0..3")

    return args


if __name__ == "__main__":
    try:
        run(parse_args())
    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
    ) as error:
        raise SystemExit(f"Error: {error}")