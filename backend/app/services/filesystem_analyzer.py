import re
import subprocess
from pathlib import Path


SUPPORTED_FILESYSTEMS = {
    "hfs": "HFS+",
    "ntfs": "NTFS",
    "fat": "FAT",
    "ext": "EXT",
}


def run_command(command: list[str]) -> str:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout


def detect_partition(image_path: Path) -> dict:
    output = run_command(
        [
            "mmls",
            str(image_path),
        ]
    )

    partition_match = None

    for line in output.splitlines():
        match = re.match(
            r"^\s*(\d+):\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(.+?)\s*$",
            line,
        )

        if not match:
            continue

        slot = match.group(1)
        partition_number = match.group(2)
        start_sector = int(match.group(3))
        end_sector = int(match.group(4))
        length = int(match.group(5))
        description = match.group(6).strip()

        if description.lower() == "unallocated":
            continue

        if description.lower() in {
            "meta",
            "safety table",
            "gpt header",
            "partition table",
        }:
            continue

        partition_match = {
            "slot": slot,
            "partition_number": partition_number,
            "start_sector": start_sector,
            "end_sector": end_sector,
            "length": length,
            "description": description,
        }

        break

    if not partition_match:
        raise ValueError(
            "Unable to identify a filesystem partition."
        )

    return {
        **partition_match,
        "raw_output": output,
    }


def detect_filesystem(
    image_path: Path,
    offset: int,
) -> dict:
    attempts = []

    for filesystem_code, filesystem_name in SUPPORTED_FILESYSTEMS.items():
        command = [
            "fsstat",
            "-f",
            filesystem_code,
            "-o",
            str(offset),
            str(image_path),
        ]

        try:
            output = run_command(command)

            return {
                "code": filesystem_code,
                "name": filesystem_name,
                "raw_output": output,
                "attempts": attempts,
            }

        except subprocess.CalledProcessError as error:
            attempts.append(
                {
                    "filesystem": filesystem_name,
                    "code": filesystem_code,
                    "error": (
                        error.stderr.strip()
                        if error.stderr
                        else "fsstat failed"
                    ),
                }
            )

    raise ValueError(
        "Unable to detect a supported filesystem."
    )


def parse_fls_output(
    file_listing: str,
    deleted: bool = False,
) -> list[dict]:
    entries = []

    for line in file_listing.splitlines():
        if not line.strip():
            continue

        match = re.match(
            r"^(?P<depth>\+*)\s*"
            r"(?P<type>[rldu])/(?P<subtype>[rldu])\s+"
            r"(?P<inode>\d+):\s*"
            r"(?P<name>.*)$",
            line,
        )

        if not match:
            continue

        depth = len(match.group("depth"))
        entry_type = match.group("type")
        inode = int(match.group("inode"))
        name = match.group("name").strip()

        entries.append(
            {
                "inode": inode,
                "type": (
                    "directory"
                    if entry_type == "d"
                    else "file"
                ),
                "name": name,
                "depth": depth,
                "deleted": deleted,
                "raw": line,
            }
        )

    return entries


def analyze_filesystem(image_path: Path) -> dict:
    partition = detect_partition(image_path)

    offset = partition["start_sector"]

    filesystem = detect_filesystem(
        image_path,
        offset,
    )

    filesystem_output = filesystem["raw_output"]

    file_listing = run_command(
        [
            "fls",
            "-f",
            filesystem["code"],
            "-o",
            str(offset),
            "-r",
            str(image_path),
        ]
    )

    entries = parse_fls_output(file_listing)

    deleted_listing = run_command(
        [
            "fls",
            "-f",
            filesystem["code"],
            "-o",
            str(offset),
            "-r",
            "-d",
            str(image_path),
        ]
    )

    deleted_entries = parse_fls_output(
        deleted_listing, deleted=True
    )

    existing_inodes = {
        entry["inode"] for entry in entries
    }

    for entry in deleted_entries:
        if entry["inode"] not in existing_inodes:
            entries.append(entry)

    volume_match = re.search(
        r"Volume Name:\s*(.+)",
        filesystem_output,
    )

    file_count_match = re.search(
        r"Number of files:\s*(\d+)",
        filesystem_output,
    )

    folder_count_match = re.search(
        r"Number of folders:\s*(\d+)",
        filesystem_output,
    )

    return {
        "partition": {
            "slot": partition["slot"],
            "partition_number": partition["partition_number"],
            "start_sector": partition["start_sector"],
            "end_sector": partition["end_sector"],
            "length": partition["length"],
            "description": partition["description"],
        },
        "filesystem": {
            "code": filesystem["code"],
            "type": filesystem["name"],
            "volume_name": (
                volume_match.group(1).strip()
                if volume_match
                else None
            ),
            "file_count": (
                int(file_count_match.group(1))
                if file_count_match
                else None
            ),
            "folder_count": (
                int(folder_count_match.group(1))
                if folder_count_match
                else None
            ),
        },
        "files": entries,
        "raw_output": {
            "mmls": partition["raw_output"],
            "fsstat": filesystem_output,
            "fls": file_listing,
        },
    }
