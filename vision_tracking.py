"""Detector adapters with motion association and current-frame-only output.

Retained tracks are used to recover identities, never as extra detections.
This module does not draw or return a missing vehicle's old bounding box.
"""
from __future__ import annotations
import copy
import json
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


class _Array:
    def __init__(self, value): self.value = np.asarray(value)
    def cpu(self): return self
    def numpy(self): return self.value


class Boxes:
    def __init__(self, xyxy, classes, confidence, ids=None):
        self.xyxy = _Array(np.asarray(xyxy, dtype=np.float32).reshape(-1, 4))
        self.cls = _Array(np.asarray(classes, dtype=int))
        self.conf = _Array(np.asarray(confidence, dtype=np.float32))
        self.id = _Array(np.asarray(ids, dtype=int)) if ids is not None else None


class DetectionResult:
    def __init__(self, boxes): self.boxes = boxes


def box_iou(a, b):
    left = np.maximum(a[:2], b[:2])
    right = np.minimum(a[2:], b[2:])
    intersection = float(np.prod(np.maximum(right - left, 0)))
    area_a = float(np.prod(np.maximum(a[2:] - a[:2], 0)))
    area_b = float(np.prod(np.maximum(b[2:] - b[:2], 0)))
    return intersection / max(area_a + area_b - intersection, 1e-8)


def suppress_duplicates(boxes, scores, threshold=0.6):
    """Class-agnostic NMS for duplicate vehicle queries in the same frame."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    order = np.argsort(-np.asarray(scores), kind='stable').tolist()
    kept = []
    while order:
        current = order.pop(0)
        kept.append(current)
        order = [j for j in order if box_iou(boxes[current], boxes[j]) <= threshold]
    return np.asarray(kept, dtype=int)


@dataclass
class Track:
    box: np.ndarray
    velocity: np.ndarray
    last_seen: float
    class_id: int
    observations: int = 1


class MotionTracker:
    """One-to-one assignment using predicted motion, distance, and overlap.

    This is a lightweight constant-velocity tracker, not a learned ReID model.
    `update` returns exactly one ID per supplied current-frame detection.
    """
    def __init__(self, max_age=0.5, center_gate=2.0):
        self.max_age = float(max_age)
        self.center_gate = float(center_gate)
        self.tracks = {}
        self.next_id = 1
        self._frame = 0

    def update(self, boxes, classes, timestamp=None):
        if timestamp is None:
            timestamp = self._frame / 30.0
        self._frame += 1
        timestamp = float(timestamp)
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        classes = np.asarray(classes, dtype=int)
        if len(boxes) != len(classes):
            raise ValueError('One class ID is required per detection')
        self.tracks = {i: r for i, r in self.tracks.items()
                       if 0 <= timestamp - r.last_seen <= self.max_age}
        track_ids = list(self.tracks)
        assigned = np.full(len(boxes), -1, dtype=int)
        cost = np.full((len(track_ids), len(boxes)), 1e6, dtype=float)
        for row, track_id in enumerate(track_ids):
            rec = self.tracks[track_id]
            predicted = rec.box + rec.velocity * max(timestamp - rec.last_seen, 0)
            predicted_size = np.maximum(predicted[2:] - predicted[:2], 1)
            for col, box in enumerate(boxes):
                size = np.maximum(box[2:] - box[:2], 1)
                scale = max(float(np.linalg.norm(predicted_size)), float(np.linalg.norm(size)), 1)
                distance = float(np.linalg.norm((predicted[:2] + predicted[2:] - box[:2] - box[2:]) / 2)) / scale
                area_ratio = max(float(np.prod(predicted_size) / np.prod(size)),
                                 float(np.prod(size) / np.prod(predicted_size)))
                if distance > self.center_gate or area_ratio > 4:
                    continue
                cost[row, col] = (0.65 * (1 - box_iou(predicted, box))
                                  + 0.35 * distance / self.center_gate
                                  + 0.05 * (rec.class_id != int(classes[col])))
        if cost.size:
            rows, cols = linear_sum_assignment(cost)
            for row, col in zip(rows, cols):
                if cost[row, col] >= 1e6:
                    continue
                track_id = track_ids[row]
                rec = self.tracks[track_id]
                dt = timestamp - rec.last_seen
                if dt > 1e-8:
                    measured = (boxes[col] - rec.box) / dt
                    rec.velocity = measured if rec.observations == 1 else 0.75 * measured + 0.25 * rec.velocity
                rec.box = boxes[col].copy()
                rec.last_seen = timestamp
                rec.class_id = int(classes[col])
                rec.observations += 1
                assigned[col] = track_id
        for col, box in enumerate(boxes):
            if assigned[col] < 0:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = Track(box.copy(), np.zeros(4, dtype=np.float32), timestamp, int(classes[col]))
                assigned[col] = track_id
        return assigned


class _Detector:
    def __init__(self, conf_threshold=0.4, nms_iou=0.6, track_max_age=0.5, track_center_gate=2.0):
        self.conf_threshold = float(conf_threshold)
        self.nms_iou = float(nms_iou)
        self._track_options = (track_max_age, track_center_gate)
        self.tracker = MotionTracker(*self._track_options)

    def fork(self):
        """Share detector weights, with a separate tracker per camera."""
        result = copy.copy(self)
        result.tracker = MotionTracker(*self._track_options)
        return result

    def _result(self, boxes, class_ids, confidences, frame, classes):
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
        class_ids = np.asarray(class_ids, dtype=int)
        confidences = np.asarray(confidences, dtype=np.float32)
        height, width = frame.shape[:2]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
        valid = (np.isfinite(boxes).all(axis=1) & np.isfinite(confidences)
                 & (confidences >= self.conf_threshold)
                 & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
        if classes is not None:
            valid &= np.isin(class_ids, list(classes))
        boxes, class_ids, confidences = boxes[valid], class_ids[valid], confidences[valid]
        keep = suppress_duplicates(boxes, confidences, self.nms_iou)
        return [DetectionResult(Boxes(boxes[keep], class_ids[keep], confidences[keep]))]

    def track(self, frame, persist=True, classes=None, verbose=False, timestamp=None):
        if not persist:
            self.tracker = MotionTracker(*self._track_options)
        result = self.predict(frame, classes=classes, verbose=verbose)[0]
        b = result.boxes
        ids = self.tracker.update(b.xyxy.numpy(), b.cls.numpy(), timestamp)
        b.id = _Array(ids)
        return [result]


class YOLODetector(_Detector):
    def __init__(self, path, device='auto', **options):
        super().__init__(**options)
        from ultralytics import YOLO
        self.model = YOLO(path)
        self.names = {int(k): str(v).lower() for k, v in self.model.names.items()}
        self.device = None if device == 'auto' else device

    def predict(self, frame, classes=None, verbose=False):
        result = self.model.predict(frame, conf=self.conf_threshold, iou=self.nms_iou,
                                    classes=list(classes) if classes is not None else None,
                                    device=self.device, verbose=False)[0]
        boxes = result.boxes
        return self._result(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy(),
                            boxes.conf.cpu().numpy(), frame, classes)


class RFDETRDetector(_Detector):
    def __init__(self, path, class_names=None, device='auto', **options):
        super().__init__(**options)
        import onnxruntime as ort
        providers = ['CPUExecutionProvider']
        if device != 'cpu' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(path), sess_options=session_options, providers=providers)
        meta = self.session.get_inputs()[0]
        if len(meta.shape) != 4 or not all(isinstance(x, int) for x in meta.shape[2:]):
            raise ValueError('This adapter requires an ONNX export with a fixed H x W input')
        self.input_name = meta.name
        self.input_h, self.input_w = meta.shape[2:]
        self.output_names = [x.name for x in self.session.get_outputs()]
        if class_names:
            self.names = {int(k): str(v).lower() for k, v in class_names.items()}
        else:
            # Preserve the old vehicle-only workflow without a speculative
            # COCO mapping. No subtype or background semantics are inferred.
            score_outputs = [x for x in self.session.get_outputs()
                             if x.name in {'labels', 'logits'}]
            if len(score_outputs) != 1 or not isinstance(score_outputs[0].shape[-1], int):
                raise ValueError('Supply --rfdetr-classes for an unknown output layout')
            self.names = {i: 'vehicle' for i in range(score_outputs[0].shape[-1])}
            print('RF-DETR: generic vehicle labels across raw score channels; '
                  'no car/van/bus subtype mapping is assumed.')
        self.mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.input_w, self.input_h)).astype(np.float32) / 255
        return np.transpose((rgb - self.mean) / self.std, (2, 0, 1))[None].astype(np.float32)

    def _decode(self, outputs, frame, classes=None):
        named = dict(zip(self.output_names, outputs))
        boxes = named.get('dets', named.get('boxes'))
        logits = named.get('labels', named.get('logits'))
        if boxes is None or logits is None:
            candidates = [x for x in outputs if x.ndim == 3 and x.shape[-1] == 4]
            others = [x for x in outputs if x.ndim == 3 and x.shape[-1] != 4]
            if len(candidates) != 1 or len(others) != 1:
                raise ValueError(f'Unknown ONNX outputs: {[x.shape for x in outputs]}')
            boxes, logits = candidates[0], others[0]
        boxes, logits = boxes[0], logits[0]
        allowed = sorted(k for k,v in self.names.items() if v not in {'background', 'no-object', '__ignore__'})
        if classes is not None:
            allowed = [k for k in allowed if k in classes]
        if not allowed:
            return self._result([], [], [], frame, classes)
        if min(allowed) < 0 or max(allowed) >= logits.shape[-1]:
            raise ValueError(f'Class mapping index outside the {logits.shape[-1]} ONNX score channels')
        scores = 1 / (1 + np.exp(-np.clip(logits[:, allowed], -80, 80)))
        chosen = scores.argmax(axis=-1)
        ids = np.asarray(allowed)[chosen]
        confidence = scores[np.arange(len(chosen)), chosen]
        h, w = frame.shape[:2]
        cx, cy, bw, bh = boxes.T
        xyxy = np.stack(((cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h), axis=1)
        return self._result(xyxy, ids, confidence, frame, classes)

    def predict(self, frame, classes=None, verbose=False):
        outputs = self.session.run(None, {self.input_name:self._preprocess(frame)})
        return self._decode(outputs, frame, classes)


def build_detector(model_path, conf_threshold=0.4, class_names_path=None,
                   nms_iou=0.6, track_max_age=0.5, track_center_gate=2.0, device='auto'):
    if not Path(model_path).is_file():
        raise FileNotFoundError(f'Detector model not found: {model_path}')
    options = dict(conf_threshold=conf_threshold, nms_iou=nms_iou,
                   track_max_age=track_max_age, track_center_gate=track_center_gate, device=device)
    suffix = Path(model_path).suffix.lower()
    if suffix == '.onnx':
        names = json.loads(Path(class_names_path).read_text()) if class_names_path else None
        return RFDETRDetector(model_path, class_names=names, **options)
    if suffix == '.pt':
        return YOLODetector(model_path, **options)
    raise ValueError('Use best.pt for YOLO or an .onnx export for RF-DETR. A PPO .zip and RF-DETR .pth are different files.')
