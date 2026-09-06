import hashlib
import json
import subprocess
from pathlib import Path


H264_SIGNATURES = (
    b"\x00\x00\x00\x01\x67",
    b"\x00\x00\x01\x67",
)

H264_START_CODES = (
    b"\x00\x00\x00\x01",
    b"\x00\x00\x01",
)


def calculate_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_media_file(file_path: Path) -> dict:
    """
    Validate a recovered media file using FFprobe.

    FFprobe accepting bytes alone is not enough.
    The recovered candidate must contain a meaningful
    H.264 video stream with valid dimensions and frame rate.
    """

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=format_name,duration,size",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate",
        "-of",
        "json",
        str(file_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return {
            "valid": False,
            "error": result.stderr.strip()
            or "ffprobe validation failed",
        }

    try:
        metadata = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {
            "valid": False,
            "error": "FFprobe returned invalid JSON.",
        }

    streams = metadata.get("streams", [])
    format_metadata = metadata.get("format", {})

    video_stream = next(
        (
            stream
            for stream in streams
            if stream.get("codec_name") == "h264"
        ),
        None,
    )

    if video_stream is None:
        return {
            "valid": False,
            "error": "No H.264 video stream detected.",
            "metadata": metadata,
        }

    width = video_stream.get("width")
    height = video_stream.get("height")
    frame_rate = video_stream.get("r_frame_rate")

    if not isinstance(width, int) or width <= 0:
        return {
            "valid": False,
            "error": "Invalid H.264 width.",
            "metadata": metadata,
        }

    if not isinstance(height, int) or height <= 0:
        return {
            "valid": False,
            "error": "Invalid H.264 height.",
            "metadata": metadata,
        }

    if not frame_rate or frame_rate in {"0", "0/0"}:
        return {
            "valid": False,
            "error": "Invalid H.264 frame rate.",
            "metadata": metadata,
        }

    try:
        numerator, denominator = frame_rate.split("/")

        fps = (
            float(numerator)
            / float(denominator)
        )

        if fps <= 0 or fps > 240:
            return {
                "valid": False,
                "error": "Implausible frame rate.",
                "metadata": metadata,
            }

    except (
        ValueError,
        ZeroDivisionError,
    ):
        return {
            "valid": False,
            "error": "Unable to parse frame rate.",
            "metadata": metadata,
        }

    return {
        "valid": True,
        "codec": "h264",
        "width": width,
        "height": height,
        "frame_rate": frame_rate,
        "format": format_metadata.get(
            "format_name"
        ),
        "size": (
            int(format_metadata["size"])
            if format_metadata.get("size")
            else None
        ),
        "duration": format_metadata.get(
            "duration"
        ),
    }


def recover_deleted_file(
    image_path: Path,
    filesystem_code: str,
    offset: int,
    inode: int,
    output_path: Path,
) -> dict:
    """
    Recover a deleted filesystem object
    using Sleuth Kit icat.
    """

    command = [
        "icat",
        "-f",
        filesystem_code,
        "-o",
        str(offset),
        "-r",
        str(image_path),
        str(inode),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        check=True,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_bytes(
        result.stdout
    )

    return {
        "method": "inode_recovery",
        "inode": inode,
        "output_path": str(output_path),
        "size": len(result.stdout),
        "sha256": calculate_sha256(
            result.stdout
        ),
        "recovered": len(result.stdout) > 0,
    }


def find_file_signatures(
    image_path: Path,
    signatures: tuple[bytes, ...],
    chunk_size: int = 1024 * 1024,
) -> list[dict]:
    """
    Search the forensic image for signatures.
    """

    if not signatures:
        return []

    matches = []

    maximum_signature_length = max(
        len(signature)
        for signature in signatures
    )

    with image_path.open("rb") as image:

        absolute_offset = 0
        previous_tail = b""

        while True:

            chunk = image.read(chunk_size)

            if not chunk:
                break

            search_data = previous_tail + chunk

            search_start = (
                absolute_offset
                - len(previous_tail)
            )

            for signature in signatures:

                position = 0

                while True:

                    position = search_data.find(
                        signature,
                        position,
                    )

                    if position == -1:
                        break

                    matches.append(
                        {
                            "offset": (
                                search_start
                                + position
                            ),
                            "signature": signature.hex(),
                        }
                    )

                    position += 1

            previous_tail = search_data[
                -(maximum_signature_length - 1):
            ]

            absolute_offset += len(chunk)

    matches.sort(
        key=lambda item: item["offset"]
    )

    return matches


def normalize_h264_signature_matches(
    matches: list[dict],
) -> list[dict]:
    """
    Collapse overlapping H.264 signatures.

    Example:

        0000000167
         00000167

    These represent one physical start.
    """

    if not matches:
        return []

    normalized = []

    matches = sorted(
        matches,
        key=lambda item: (
            item["offset"],
            -len(bytes.fromhex(
                item["signature"]
            )),
        ),
    )

    for match in matches:

        if not normalized:
            normalized.append(match)
            continue

        previous = normalized[-1]

        previous_offset = previous["offset"]
        previous_length = len(
            bytes.fromhex(
                previous["signature"]
            )
        )

        previous_end = (
            previous_offset
            + previous_length
        )

        current_offset = match["offset"]

        if current_offset < previous_end:
            continue

        normalized.append(match)

    return normalized


def find_h264_signatures(
    image_path: Path,
) -> list[dict]:

    matches = find_file_signatures(
        image_path=image_path,
        signatures=H264_SIGNATURES,
    )

    return normalize_h264_signature_matches(
        matches
    )


def find_h264_nal_units(
    image_path: Path,
    start_offset: int,
    max_scan_size: int = 64 * 1024 * 1024,
) -> list[dict]:
    """
    Lightweight Annex-B NAL parser.
    """

    with image_path.open("rb") as image:
        image.seek(start_offset)
        data = image.read(max_scan_size)

    units = []
    positions = []

    search_position = 0

    while search_position < len(data):

        found_position = None
        found_length = None

        for start_code in H264_START_CODES:

            position = data.find(
                start_code,
                search_position,
            )

            if position == -1:
                continue

            if (
                found_position is None
                or position < found_position
            ):
                found_position = position
                found_length = len(start_code)

        if found_position is None:
            break

        positions.append(
            (
                found_position,
                found_length,
            )
        )

        search_position = (
            found_position
            + found_length
        )

    for index, (
        position,
        length,
    ) in enumerate(positions):

        header_position = position + length

        if header_position >= len(data):
            continue

        nal_type = (
            data[header_position] & 0x1F
        )

        if index + 1 < len(positions):
            end = positions[index + 1][0]
        else:
            end = len(data)

        units.append(
            {
                "relative_offset": position,
                "absolute_offset": (
                    start_offset + position
                ),
                "nal_type": nal_type,
                "size": end - position,
            }
        )

    return units


def detect_h264_boundary(
    image_path: Path,
    start_offset: int,
    max_scan_size: int = 64 * 1024 * 1024,
) -> dict:
    """
    Structural analysis only.

    Never claims an exact recording boundary.
    """

    units = find_h264_nal_units(
        image_path=image_path,
        start_offset=start_offset,
        max_scan_size=max_scan_size,
    )

    counts = {}

    for unit in units:
        nal_type = unit["nal_type"]

        counts[nal_type] = (
            counts.get(nal_type, 0)
            + 1
        )

    return {
        "boundary_found": False,
        "confidence": "UNCERTAIN",
        "reason": (
            "Generic H.264 NAL structure alone "
            "cannot establish the original "
            "recording boundary."
        ),
        "start_offset": start_offset,
        "end_offset": None,
        "size": None,
        "nal_units": len(units),
        "nal_type_counts": counts,
    }


def detect_zero_tail_boundary(
    image_path: Path,
    start_offset: int,
    max_scan_size: int = 64 * 1024 * 1024,
    zero_run_threshold: int = 512,
    minimum_content_size: int = 1024,
) -> dict:
    """
    Detect a sustained zero-filled region.

    The beginning of the first sufficiently long
    zero run becomes a boundary candidate.

    This does NOT establish an exact forensic
    boundary. It only provides medium-confidence
    evidence for a contiguous candidate.
    """

    with image_path.open("rb") as image:
        image.seek(start_offset)
        data = image.read(max_scan_size)

    zero_run = 0

    for index, byte in enumerate(data):

        if byte == 0:
            zero_run += 1
        else:
            zero_run = 0

        if zero_run < zero_run_threshold:
            continue

        boundary = (
            index
            - zero_run
            + 1
        )

        if boundary < minimum_content_size:
            continue

        return {
            "boundary_found": True,
            "confidence": "MEDIUM",
            "method": "sustained_zero_tail",
            "start_offset": start_offset,
            "end_offset": (
                start_offset + boundary
            ),
            "size": boundary,
            "zero_run": zero_run_threshold,
            "exact_original_boundary_established": False,
        }

    return {
        "boundary_found": False,
        "confidence": "UNCERTAIN",
        "method": None,
        "start_offset": start_offset,
        "end_offset": None,
        "size": None,
        "zero_run": 0,
        "exact_original_boundary_established": False,
    }


def carve_contiguous_file(
    image_path: Path,
    offset: int,
    size: int,
    output_path: Path,
) -> dict:

    if offset < 0:
        raise ValueError(
            "Carving offset cannot be negative."
        )

    if size <= 0:
        raise ValueError(
            "Carving size must be greater than zero."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with image_path.open("rb") as image:

        image.seek(offset)

        data = image.read(size)

    if len(data) != size:
        raise ValueError(
            "Insufficient bytes available "
            "for requested carve."
        )

    output_path.write_bytes(data)

    return {
        "method": "contiguous_carving",
        "image_offset": offset,
        "output_path": str(output_path),
        "size": len(data),
        "sha256": calculate_sha256(data),
        "recovered": True,
    }


def build_uncertain_recovery_result(
    image_path: Path,
    offset: int,
    output_path: Path,
    size: int,
) -> dict:

    result = carve_contiguous_file(
        image_path=image_path,
        offset=offset,
        size=size,
        output_path=output_path,
    )

    validation = validate_media_file(
        output_path
    )

    result["validation"] = validation

    if not validation["valid"]:
        output_path.unlink(
            missing_ok=True
        )

        raise ValueError(
            "Candidate failed validation."
        )

    result["boundary"] = {
        "confidence": "UNCERTAIN",
        "method": "externally_supplied_candidate_size",
        "candidate_start_offset": offset,
        "candidate_size": size,
        "exact_original_boundary_established": False,
    }

    result["recovery_status"] = (
        "VALIDATED_CANDIDATE"
    )

    return result


def carve_h264_file(
    image_path: Path,
    size: int,
    output_path: Path,
) -> dict:

    matches = find_h264_signatures(
        image_path
    )

    if not matches:
        raise FileNotFoundError(
            "No H.264 signature found."
        )

    for match in matches:

        try:
            result = build_uncertain_recovery_result(
                image_path=image_path,
                offset=match["offset"],
                output_path=output_path,
                size=size,
            )

            result["signature"] = (
                match["signature"]
            )

            result["boundary_analysis"] = (
                detect_h264_boundary(
                    image_path=image_path,
                    start_offset=match["offset"],
                )
            )

            return result

        except ValueError:
            continue

    raise ValueError(
        "No valid H.264 candidate found."
    )


def carve_h264_candidates(
    image_path: Path,
    output_directory: Path,
    max_candidate_size: int = 64 * 1024 * 1024,
    max_candidates: int = 20,
    zero_run_threshold: int = 512,
    minimum_candidate_size: int = 1024,
) -> list[dict]:
    """
    Automatic H.264 carving.

    Strategy:

    1. Detect H.264 start.
    2. Normalize overlapping signatures.
    3. Detect sustained zero tail.
    4. Carve until candidate boundary.
    5. Validate with FFprobe.
    6. Record MEDIUM confidence.

    If no zero-tail boundary exists,
    the candidate is rejected instead of
    pretending a maximum scan window is
    the recovered file.
    """

    matches = find_h264_signatures(
        image_path
    )

    if not matches:
        return []

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for match in matches[:max_candidates]:

        start_offset = match["offset"]

        boundary = detect_zero_tail_boundary(
            image_path=image_path,
            start_offset=start_offset,
            max_scan_size=max_candidate_size,
            zero_run_threshold=zero_run_threshold,
            minimum_content_size=minimum_candidate_size,
        )

        if not boundary["boundary_found"]:
            continue

        candidate_size = boundary["size"]

        if candidate_size < minimum_candidate_size:
            continue

        output_path = (
            output_directory
            / f"carved_h264_{start_offset}.h264"
        )

        try:
            result = carve_contiguous_file(
                image_path=image_path,
                offset=start_offset,
                size=candidate_size,
                output_path=output_path,
            )

            validation = validate_media_file(
                output_path
            )

            if not validation["valid"]:
                output_path.unlink(
                    missing_ok=True
                )
                continue

            result["validation"] = validation
            result["signature"] = (
                match["signature"]
            )

            result["boundary"] = boundary

            result["recovery_status"] = (
                "VALIDATED_CANDIDATE"
            )

            results.append(result)

        except (
            OSError,
            ValueError,
        ):
            output_path.unlink(
                missing_ok=True
            )

    return results