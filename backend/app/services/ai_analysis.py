from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.models.ai_observation import (
    AIAnalysisResult,
    AICapabilitiesResult,
    AICapability,
    AIObservation,
    BoundingBox,
    TimeMapping,
    MOTION,
    OBJECT_DETECTION,
    FACE_DETECTION,
    NOT_AVAILABLE,
    COMPLETED,
    FAILED,
)
from app.services.ai_providers import (
    get_provider,
    get_available_providers,
)


def get_capabilities() -> AICapabilitiesResult:
    available = get_available_providers()

    motion_cap = AICapability(
        analysis_type=MOTION,
        available=available.get("motion", False),
        provider="opencv_motion" if available.get("motion") else None,
        model_name="OpenCV" if available.get("motion") else None,
        reason=None if available.get("motion") else "OpenCV not installed",
    )

    obj_provider = get_provider("object_detection")
    obj_cap = AICapability(
        analysis_type=OBJECT_DETECTION,
        available=available.get("object_detection", False),
        provider=obj_provider.provider_name if obj_provider and obj_provider.is_available else None,
        model_name=obj_provider.model_name if obj_provider and obj_provider.is_available else None,
        model_version=obj_provider.model_version if obj_provider and obj_provider.is_available else None,
        reason=None if available.get("object_detection") else "NanoDet model not installed or not loadable",
    )

    face_provider = get_provider("face_detection")
    face_cap = AICapability(
        analysis_type=FACE_DETECTION,
        available=available.get("face_detection", False),
        provider=face_provider.provider_name if face_provider and face_provider.is_available else None,
        model_name=face_provider.model_name if face_provider and face_provider.is_available else None,
        model_version=face_provider.model_version if face_provider and face_provider.is_available else None,
        reason=None if available.get("face_detection") else "YuNet model not installed or not loadable",
    )

    return AICapabilitiesResult(
        motion=motion_cap,
        object_detection=obj_cap,
        face_detection=face_cap,
    )


def _normalize_observation(
    raw: dict,
    evidence_id: str | None = None,
    recording_filename: str | None = None,
    camera_id: str | None = None,
    provider_name: str | None = None,
    analysis_type: str = "",
    start_time: str | None = None,
) -> AIObservation | None:
    if raw.get("status") in ("NOT_AVAILABLE", "FAILED"):
        return None

    label = raw.get("label")
    if not label:
        return None

    time_mapping = None
    tm_data = raw.get("time_mapping")
    if tm_data:
        time_mapping = TimeMapping(
            media_offset_seconds=tm_data.get("media_offset_seconds"),
            source_recording_start=tm_data.get("source_recording_start"),
            normalized_timestamp=tm_data.get("normalized_timestamp"),
            timezone=tm_data.get("timezone"),
            mapping_method=tm_data.get("mapping_method"),
            mapping_status=tm_data.get("mapping_status", "MAPPING_UNCERTAIN"),
        )

    bbox_data = raw.get("bounding_box")
    bounding_box = None
    if bbox_data:
        bounding_box = BoundingBox(
            x=bbox_data.get("x", 0),
            y=bbox_data.get("y", 0),
            width=bbox_data.get("width", 0),
            height=bbox_data.get("height", 0),
        )

    confidence = raw.get("confidence")
    if confidence is not None:
        try:
            confidence = float(confidence)
            if not (0.0 <= confidence <= 1.0):
                confidence = None
        except (ValueError, TypeError):
            confidence = None

    uncertainty = raw.get("uncertainty")
    if time_mapping and time_mapping.mapping_status == "MAPPING_UNCERTAIN":
        uncertainty = uncertainty or "timestamp mapping uncertain"

    return AIObservation(
        observation_id=f"OBS-{uuid4().hex[:8].upper()}",
        evidence_id=evidence_id,
        recording_filename=recording_filename,
        camera_id=camera_id,
        analysis_type=analysis_type,
        label=label,
        confidence=confidence,
        frame_number=raw.get("frame_number"),
        media_timestamp=raw.get("media_timestamp"),
        time_mapping=time_mapping,
        bounding_box=bounding_box,
        source_media_path=raw.get("source_media_path"),
        provider=provider_name,
        model_name=raw.get("model_name"),
        model_version=raw.get("model_version"),
        status=COMPLETED,
        uncertainty=uncertainty,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def analyze_video_file(
    video_path: Path,
    analysis_type: str,
    evidence_id: str | None = None,
    recording_filename: str | None = None,
    camera_id: str | None = None,
    start_time: str | None = None,
    sample_interval_seconds: float = 1.0,
    max_frames: int | None = None,
) -> AIAnalysisResult:
    started_at = datetime.now(timezone.utc).isoformat()

    provider = get_provider(analysis_type)
    if not provider:
        return AIAnalysisResult(
            evidence_id=evidence_id,
            recording_filename=recording_filename,
            camera_id=camera_id,
            analysis_type=analysis_type,
            status=NOT_AVAILABLE,
            error=f"No provider registered for analysis type: {analysis_type}",
            started_at=started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

    if not provider.is_available:
        return AIAnalysisResult(
            evidence_id=evidence_id,
            recording_filename=recording_filename,
            camera_id=camera_id,
            analysis_type=analysis_type,
            provider=provider.provider_name,
            status=NOT_AVAILABLE,
            error=f"Provider {provider.provider_name} is not available",
            started_at=started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

    if not video_path.exists():
        return AIAnalysisResult(
            evidence_id=evidence_id,
            recording_filename=recording_filename,
            camera_id=camera_id,
            analysis_type=analysis_type,
            provider=provider.provider_name,
            status=FAILED,
            error=f"Video file not found: {video_path}",
            started_at=started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

    try:
        raw_observations = provider.analyze_video(
            video_path=video_path,
            evidence_id=evidence_id,
            recording_filename=recording_filename,
            camera_id=camera_id,
            start_time=start_time,
            sample_interval_seconds=sample_interval_seconds,
            max_frames=max_frames,
        )
    except Exception as e:
        return AIAnalysisResult(
            evidence_id=evidence_id,
            recording_filename=recording_filename,
            camera_id=camera_id,
            analysis_type=analysis_type,
            provider=provider.provider_name,
            status=FAILED,
            error=str(e),
            started_at=started_at,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

    observations = []
    uncertainty_notes = []

    for raw in raw_observations:
        if raw.get("status") in ("NOT_AVAILABLE", "FAILED"):
            if raw.get("error"):
                uncertainty_notes.append(raw["error"])
            continue

        obs = _normalize_observation(
            raw,
            evidence_id=evidence_id,
            recording_filename=recording_filename,
            camera_id=camera_id,
            provider_name=provider.provider_name,
            analysis_type=analysis_type,
            start_time=start_time,
        )
        if obs:
            observations.append(obs)
            if obs.uncertainty:
                uncertainty_notes.append(obs.uncertainty)

    completed_at = datetime.now(timezone.utc).isoformat()

    status = COMPLETED
    if not observations and uncertainty_notes:
        status = PARTIAL
    elif not observations:
        status = COMPLETED

    return AIAnalysisResult(
        evidence_id=evidence_id,
        recording_filename=recording_filename,
        camera_id=camera_id,
        analysis_type=analysis_type,
        provider=provider.provider_name,
        status=status,
        observations=observations,
        observation_count=len(observations),
        started_at=started_at,
        completed_at=completed_at,
        uncertainty_notes=uncertainty_notes,
    )


PARTIAL = "PARTIAL"


def analyze_media(
    media_path: Path,
    evidence_id: str | None = None,
    recording_filename: str | None = None,
    camera_id: str | None = None,
    start_time: str | None = None,
    analysis_types: list[str] | None = None,
    sample_interval_seconds: float = 1.0,
    max_frames: int | None = None,
) -> dict[str, AIAnalysisResult]:
    if analysis_types is None:
        analysis_types = [MOTION, OBJECT_DETECTION, FACE_DETECTION]

    results = {}
    for atype in analysis_types:
        results[atype] = analyze_video_file(
            video_path=media_path,
            analysis_type=atype,
            evidence_id=evidence_id,
            recording_filename=recording_filename,
            camera_id=camera_id,
            start_time=start_time,
            sample_interval_seconds=sample_interval_seconds,
            max_frames=max_frames,
        )

    return results


def filter_observations(
    observations: list[AIObservation],
    evidence_id: str | None = None,
    camera_id: str | None = None,
    recording_filename: str | None = None,
    analysis_type: str | None = None,
    label: str | None = None,
    min_confidence: float | None = None,
    provider: str | None = None,
) -> list[AIObservation]:
    result = list(observations)

    if evidence_id:
        result = [o for o in result if o.evidence_id == evidence_id]
    if camera_id:
        result = [o for o in result if o.camera_id == camera_id]
    if recording_filename:
        result = [o for o in result if o.recording_filename == recording_filename]
    if analysis_type:
        result = [o for o in result if o.analysis_type == analysis_type]
    if label:
        result = [o for o in result if o.label == label]
    if min_confidence is not None:
        result = [o for o in result if o.confidence is not None and o.confidence >= min_confidence]
    if provider:
        result = [o for o in result if o.provider == provider]

    result.sort(key=lambda o: (o.media_timestamp or "", o.observation_id))
    return result
