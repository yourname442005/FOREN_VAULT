from abc import ABC, abstractmethod
from pathlib import Path


MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
NANODET_MODEL = MODELS_DIR / "object_detection_nanodet_2022nov.onnx"

COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


class VideoAnalysisProvider(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    @property
    @abstractmethod
    def analysis_type(self) -> str:
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        ...

    @abstractmethod
    def analyze_video(
        self,
        video_path: Path,
        evidence_id: str | None = None,
        recording_filename: str | None = None,
        camera_id: str | None = None,
        start_time: str | None = None,
        sample_interval_seconds: float = 1.0,
        max_frames: int | None = None,
    ) -> list[dict]:
        ...

    @abstractmethod
    def analyze_frame(
        self,
        frame,
        evidence_id: str | None = None,
        recording_filename: str | None = None,
        camera_id: str | None = None,
        frame_number: int | None = None,
        media_timestamp: float | None = None,
    ) -> list[dict]:
        ...

    def get_capabilities(self) -> dict:
        return {
            "provider": self.provider_name,
            "analysis_type": self.analysis_type,
            "available": self.is_available,
        }


class MotionProvider(VideoAnalysisProvider):
    @property
    def provider_name(self) -> str:
        return "opencv_motion"

    @property
    def analysis_type(self) -> str:
        return "motion"

    @property
    def is_available(self) -> bool:
        try:
            import cv2
            return cv2 is not None
        except ImportError:
            return False

    def analyze_video(
        self,
        video_path: Path,
        evidence_id: str | None = None,
        recording_filename: str | None = None,
        camera_id: str | None = None,
        start_time: str | None = None,
        sample_interval_seconds: float = 1.0,
        max_frames: int | None = None,
    ) -> list[dict]:
        if not self.is_available:
            return [{"status": "NOT_AVAILABLE", "error": "OpenCV not installed"}]

        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return [{"status": "FAILED", "error": f"Cannot open video: {video_path}"}]

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval = max(1, int(fps * sample_interval_seconds))

        observations = []
        prev_gray = None
        frame_idx = 0
        analyzed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and analyzed >= max_frames:
                break

            if frame_idx % frame_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                if prev_gray is not None:
                    delta = cv2.absdiff(prev_gray, gray)
                    thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
                    thresh = cv2.dilate(thresh, None, iterations=2)
                    motion_pixels = cv2.countNonZero(thresh)
                    total_pixels = thresh.shape[0] * thresh.shape[1]
                    motion_ratio = motion_pixels / total_pixels if total_pixels > 0 else 0.0

                    media_time = frame_idx / fps
                    media_offset = media_time

                    normalized_ts = None
                    mapping_status = "MAPPING_UNCERTAIN"
                    if start_time:
                        try:
                            from datetime import datetime, timedelta
                            start_dt = datetime.fromisoformat(start_time)
                            obs_dt = start_dt + timedelta(seconds=media_time)
                            normalized_ts = obs_dt.isoformat()
                            mapping_status = "MAPPED"
                        except (ValueError, TypeError):
                            pass

                    observations.append({
                        "label": "motion_detected" if motion_ratio > 0.01 else "no_motion",
                        "confidence": min(1.0, motion_ratio * 10),
                        "frame_number": frame_idx,
                        "media_timestamp": f"{media_time:.3f}",
                        "time_mapping": {
                            "media_offset_seconds": media_offset,
                            "source_recording_start": start_time,
                            "normalized_timestamp": normalized_ts,
                            "mapping_status": mapping_status,
                        },
                        "motion_ratio": round(motion_ratio, 6),
                        "motion_pixels": motion_pixels,
                        "total_pixels": total_pixels,
                    })

                prev_gray = gray
                analyzed += 1

            frame_idx += 1

        cap.release()
        return observations

    def analyze_frame(
        self,
        frame,
        evidence_id: str | None = None,
        recording_filename: str | None = None,
        camera_id: str | None = None,
        frame_number: int | None = None,
        media_timestamp: float | None = None,
    ) -> list[dict]:
        if not self.is_available:
            return [{"status": "NOT_AVAILABLE"}]

        import cv2

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        return [{
            "label": "frame_analyzed",
            "confidence": 0.0,
            "frame_number": frame_number,
            "media_timestamp": media_timestamp,
            "status": "COMPLETED",
        }]


class FaceDetectionProvider(VideoAnalysisProvider):
    def __init__(self, confidence_threshold: float = 0.5):
        self._confidence_threshold = confidence_threshold
        self._detector = None
        self._loaded = False

    @property
    def provider_name(self) -> str:
        return "yunet"

    @property
    def analysis_type(self) -> str:
        return "face_detection"

    @property
    def model_name(self) -> str:
        return "face_detection_yunet_2023mar"

    @property
    def model_version(self) -> str:
        return "2023mar"

    def _load_model(self):
        if self._loaded:
            return
        try:
            import cv2
            if not YUNET_MODEL.exists():
                self._loaded = True
                return
            self._detector = cv2.FaceDetectorYN.create(
                str(YUNET_MODEL),
                "",
                (320, 320),
                score_threshold=self._confidence_threshold,
                nms_threshold=0.3,
                top_k=5000,
            )
            self._loaded = True
        except Exception:
            self._loaded = True
            self._detector = None

    @property
    def is_available(self) -> bool:
        self._load_model()
        return self._detector is not None

    def _detect_faces(self, frame):
        if not self.is_available:
            return []


        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))
        _, detections = self._detector.detect(frame)

        if detections is None:
            return []

        results = []
        for det in detections:
            x, y, w_box, h_box = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            confidence = float(det[4])
            results.append({
                "x": x,
                "y": y,
                "width": w_box,
                "height": h_box,
                "confidence": confidence,
            })
        return results

    def analyze_video(
        self,
        video_path: Path,
        evidence_id: str | None = None,
        recording_filename: str | None = None,
        camera_id: str | None = None,
        start_time: str | None = None,
        sample_interval_seconds: float = 1.0,
        max_frames: int | None = None,
    ) -> list[dict]:
        if not self.is_available:
            return [{"status": "NOT_AVAILABLE", "error": "YuNet model not loaded"}]

        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return [{"status": "FAILED", "error": f"Cannot open video: {video_path}"}]

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval = max(1, int(fps * sample_interval_seconds))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        observations = []
        frame_idx = 0
        analyzed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and analyzed >= max_frames:
                break

            if frame_idx % frame_interval == 0:
                detections = self._detect_faces(frame)

                media_time = frame_idx / fps
                normalized_ts = None
                mapping_status = "MAPPING_UNCERTAIN"
                if start_time:
                    try:
                        from datetime import datetime, timedelta
                        start_dt = datetime.fromisoformat(start_time)
                        obs_dt = start_dt + timedelta(seconds=media_time)
                        normalized_ts = obs_dt.isoformat()
                        mapping_status = "MAPPED"
                    except (ValueError, TypeError):
                        pass

                for det in detections:
                    observations.append({
                        "label": "face",
                        "confidence": det["confidence"],
                        "frame_number": frame_idx,
                        "media_timestamp": f"{media_time:.3f}",
                        "bounding_box": {
                            "x": det["x"],
                            "y": det["y"],
                            "width": det["width"],
                            "height": det["height"],
                        },
                        "time_mapping": {
                            "media_offset_seconds": media_time,
                            "source_recording_start": start_time,
                            "normalized_timestamp": normalized_ts,
                            "mapping_status": mapping_status,
                        },
                        "model_name": self.model_name,
                        "model_version": self.model_version,
                        "frame_dimensions": {"width": frame_w, "height": frame_h},
                    })

                analyzed += 1

            frame_idx += 1

        cap.release()
        return observations

    def analyze_frame(
        self,
        frame,
        evidence_id: str | None = None,
        recording_filename: str | None = None,
        camera_id: str | None = None,
        frame_number: int | None = None,
        media_timestamp: float | None = None,
    ) -> list[dict]:
        if not self.is_available:
            return [{"status": "NOT_AVAILABLE"}]

        detections = self._detect_faces(frame)

        observations = []
        for det in detections:
            observations.append({
                "label": "face",
                "confidence": det["confidence"],
                "frame_number": frame_number,
                "media_timestamp": media_timestamp,
                "bounding_box": {
                    "x": det["x"],
                    "y": det["y"],
                    "width": det["width"],
                    "height": det["height"],
                },
                "model_name": self.model_name,
                "model_version": self.model_version,
            })

        if not observations:
            observations.append({
                "label": "no_face",
                "confidence": 0.0,
                "frame_number": frame_number,
                "media_timestamp": media_timestamp,
                "model_name": self.model_name,
                "model_version": self.model_version,
            })

        return observations


class ObjectDetectionProvider(VideoAnalysisProvider):
    def __init__(self, confidence_threshold: float = 0.4, nms_threshold: float = 0.45):
        self._confidence_threshold = confidence_threshold
        self._nms_threshold = nms_threshold
        self._net = None
        self._loaded = False

    @property
    def provider_name(self) -> str:
        return "nanodet"

    @property
    def analysis_type(self) -> str:
        return "object_detection"

    @property
    def model_name(self) -> str:
        return "object_detection_nanodet_2022nov"

    @property
    def model_version(self) -> str:
        return "2022nov"

    def _load_model(self):
        if self._loaded:
            return
        try:
            import cv2
            if not NANODET_MODEL.exists():
                self._loaded = True
                return
            self._net = cv2.dnn.readNetFromONNX(str(NANODET_MODEL))
            self._loaded = True
        except Exception:
            self._loaded = True
            self._net = None

    @property
    def is_available(self) -> bool:
        self._load_model()
        return self._net is not None

    def _preprocess(self, frame, input_size=(416, 416)):
        import cv2
        blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, input_size, swapRB=True, crop=False)
        return blob

    def _postprocess(self, output, frame_shape, confidence_threshold):
        import numpy as np

        h, w = frame_shape[:2]

        if len(output.shape) == 3:
            output = output[0]

        detections = []
        for det in output:
            if len(det) < 5:
                continue
            scores = det[4:]
            if len(scores) == 0:
                continue
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence < confidence_threshold:
                continue

            cx, cy, bw, bh = det[0], det[1], det[2], det[3]
            x = int((cx - bw / 2) * w)
            y = int((cy - bh / 2) * h)
            width = int(bw * w)
            height = int(bh * h)

            x = max(0, x)
            y = max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)

            if width > 0 and height > 0:
                detections.append({
                    "class_id": class_id,
                    "label": COCO_LABELS[class_id] if class_id < len(COCO_LABELS) else f"class_{class_id}",
                    "confidence": confidence,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                })

        return detections

    def analyze_video(
        self,
        video_path: Path,
        evidence_id: str | None = None,
        recording_filename: str | None = None,
        camera_id: str | None = None,
        start_time: str | None = None,
        sample_interval_seconds: float = 1.0,
        max_frames: int | None = None,
    ) -> list[dict]:
        if not self.is_available:
            return [{"status": "NOT_AVAILABLE", "error": "NanoDet model not loaded"}]

        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return [{"status": "FAILED", "error": f"Cannot open video: {video_path}"}]

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval = max(1, int(fps * sample_interval_seconds))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        observations = []
        frame_idx = 0
        analyzed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and analyzed >= max_frames:
                break

            if frame_idx % frame_interval == 0:
                blob = self._preprocess(frame)
                self._net.setInput(blob)
                output = self._net.forward()
                detections = self._postprocess(output, frame.shape, self._confidence_threshold)

                media_time = frame_idx / fps
                normalized_ts = None
                mapping_status = "MAPPING_UNCERTAIN"
                if start_time:
                    try:
                        from datetime import datetime, timedelta
                        start_dt = datetime.fromisoformat(start_time)
                        obs_dt = start_dt + timedelta(seconds=media_time)
                        normalized_ts = obs_dt.isoformat()
                        mapping_status = "MAPPED"
                    except (ValueError, TypeError):
                        pass

                for det in detections:
                    observations.append({
                        "label": det["label"],
                        "confidence": det["confidence"],
                        "frame_number": frame_idx,
                        "media_timestamp": f"{media_time:.3f}",
                        "bounding_box": {
                            "x": det["x"],
                            "y": det["y"],
                            "width": det["width"],
                            "height": det["height"],
                        },
                        "class_id": det["class_id"],
                        "time_mapping": {
                            "media_offset_seconds": media_time,
                            "source_recording_start": start_time,
                            "normalized_timestamp": normalized_ts,
                            "mapping_status": mapping_status,
                        },
                        "model_name": self.model_name,
                        "model_version": self.model_version,
                        "frame_dimensions": {"width": frame_w, "height": frame_h},
                    })

                analyzed += 1

            frame_idx += 1

        cap.release()
        return observations

    def analyze_frame(
        self,
        frame,
        evidence_id: str | None = None,
        recording_filename: str | None = None,
        camera_id: str | None = None,
        frame_number: int | None = None,
        media_timestamp: float | None = None,
    ) -> list[dict]:
        if not self.is_available:
            return [{"status": "NOT_AVAILABLE"}]

        blob = self._preprocess(frame)
        self._net.setInput(blob)
        output = self._net.forward()
        detections = self._postprocess(output, frame.shape, self._confidence_threshold)

        observations = []
        for det in detections:
            observations.append({
                "label": det["label"],
                "confidence": det["confidence"],
                "frame_number": frame_number,
                "media_timestamp": media_timestamp,
                "bounding_box": {
                    "x": det["x"],
                    "y": det["y"],
                    "width": det["width"],
                    "height": det["height"],
                },
                "class_id": det["class_id"],
                "model_name": self.model_name,
                "model_version": self.model_version,
            })

        if not observations:
            observations.append({
                "label": "no_object",
                "confidence": 0.0,
                "frame_number": frame_number,
                "media_timestamp": media_timestamp,
                "model_name": self.model_name,
                "model_version": self.model_version,
            })

        return observations


PROVIDER_REGISTRY: dict[str, type[VideoAnalysisProvider]] = {
    "motion": MotionProvider,
    "object_detection": ObjectDetectionProvider,
    "face_detection": FaceDetectionProvider,
}


def get_provider(analysis_type: str) -> VideoAnalysisProvider | None:
    provider_cls = PROVIDER_REGISTRY.get(analysis_type)
    if provider_cls:
        return provider_cls()
    return None


def get_available_providers() -> dict[str, bool]:
    return {
        name: cls().is_available
        for name, cls in PROVIDER_REGISTRY.items()
    }
