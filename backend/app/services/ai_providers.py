from abc import ABC, abstractmethod
from pathlib import Path


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


class ObjectDetectionProvider(VideoAnalysisProvider):
    @property
    def provider_name(self) -> str:
        return "unavailable_object_detector"

    @property
    def analysis_type(self) -> str:
        return "object_detection"

    @property
    def is_available(self) -> bool:
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
        return [{
            "status": "NOT_AVAILABLE",
            "error": "No object detection model available. "
                     "Install a compatible object detection provider to enable this capability.",
        }]

    def analyze_frame(
        self,
        frame,
        evidence_id: str | None = None,
        recording_filename: str | None = None,
        camera_id: str | None = None,
        frame_number: int | None = None,
        media_timestamp: float | None = None,
    ) -> list[dict]:
        return [{"status": "NOT_AVAILABLE"}]


class FaceDetectionProvider(VideoAnalysisProvider):
    @property
    def provider_name(self) -> str:
        return "unavailable_face_detector"

    @property
    def analysis_type(self) -> str:
        return "face_detection"

    @property
    def is_available(self) -> bool:
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
        return [{
            "status": "NOT_AVAILABLE",
            "error": "No face detection model available. "
                     "Install a compatible face detection provider to enable this capability.",
        }]

    def analyze_frame(
        self,
        frame,
        evidence_id: str | None = None,
        recording_filename: str | None = None,
        camera_id: str | None = None,
        frame_number: int | None = None,
        media_timestamp: float | None = None,
    ) -> list[dict]:
        return [{"status": "NOT_AVAILABLE"}]


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
