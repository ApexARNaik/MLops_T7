"""MOOCCubeX dataset downloader for Feature A (Hobby-based learning).

Downloads relations/concept-problem.txt completely and streams only the first
200 MiB of relations/user-problem.json from the official MOOCCubeX repository,
with support for resuming interrupted downloads via HTTP Range headers.
"""

from pathlib import Path
import sys
import time
import requests

BASE_URL: str = "https://lfs.aminer.cn/misc/moocdata/data/mooccube2"
RAW_DIR: Path = Path("data/hobby_learning/raw")
TARGET_SAMPLE_BYTES: int = 200 * 1024 * 1024  # 200 MiB = 209,715,200 bytes
CHUNK_SIZE: int = 1024 * 1024  # 1 MiB chunk streaming


def download_stream(
    url: str,
    output_path: Path,
    limit_bytes: int | None = None,
    timeout: int = 60,
    max_retries: int = 5,
) -> None:
    """Stream download a file from a URL to an output path with optional byte limit

    and resume capability for network drops.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    retries = 0
    while retries < max_retries:
        existing_size = output_path.stat().st_size if output_path.exists() else 0

        if limit_bytes is not None and existing_size >= limit_bytes:
            print(
                f"[SKIP] {output_path} already has {existing_size:,} bytes "
                f"(>= requested limit {limit_bytes:,} bytes)."
            )
            return

        if limit_bytes is None and existing_size > 0 and retries == 0:
            # Check with a HEAD request if full file is already there
            try:
                head_resp = requests.head(url, timeout=30)
                cl = head_resp.headers.get("Content-Length")
                if cl and existing_size >= int(cl):
                    print(f"[SKIP] {output_path} complete ({existing_size:,} bytes).")
                    return
            except Exception:
                pass

        headers = {}
        file_mode = "wb"
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            file_mode = "ab"
            print(f"\n[RESUME] Resuming from byte {existing_size:,}...")
        else:
            print(f"\n[DOWNLOAD] Requesting: {url}")

        try:
            with requests.get(
                url, stream=True, headers=headers, timeout=timeout
            ) as response:
                print(f"HTTP Status: {response.status_code}")
                print(
                    f"Content-Type: {response.headers.get('Content-Type', 'unknown')}"
                )
                if response.status_code not in (200, 206):
                    response.raise_for_status()

                # If server doesn't support Range (returns 200 instead of 206), start over
                if existing_size > 0 and response.status_code == 200:
                    print(
                        "[WARN] Server did not accept Range header; restarting download."
                    )
                    existing_size = 0
                    file_mode = "wb"

                bytes_written = existing_size
                with open(output_path, file_mode) as f:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        if limit_bytes is not None:
                            remaining = limit_bytes - bytes_written
                            if remaining <= 0:
                                break
                            if len(chunk) > remaining:
                                chunk = chunk[:remaining]

                        f.write(chunk)
                        bytes_written += len(chunk)

                        if limit_bytes is not None and bytes_written >= limit_bytes:
                            print(
                                f"Reached download limit of {limit_bytes:,} bytes "
                                f"({limit_bytes / (1024 * 1024):.2f} MiB). Terminating stream."
                            )
                            return

                print(
                    f"Finished saving to {output_path} ({bytes_written:,} bytes / "
                    f"{bytes_written / (1024 * 1024):.2f} MiB written)."
                )
                return

        except (
            requests.RequestException,
            ConnectionError,
            ConnectionResetError,
        ) as exc:
            retries += 1
            print(
                f"[WARN] Connection dropped: {exc}. Retrying ({retries}/{max_retries}) in 3s...",
                file=sys.stderr,
            )
            time.sleep(3)

    raise RuntimeError(
        f"Failed to complete download after {max_retries} retries for {url}"
    )


def inspect_file(file_path: Path, count_newlines: bool = False) -> None:
    """Print file size, first 500 characters, and optionally count newline characters."""
    if not file_path.exists():
        print(f"[ERROR] Cannot inspect {file_path}: file does not exist.")
        return

    file_size = file_path.stat().st_size
    print(f"\n--- Inspection: {file_path} ---")
    print(f"File Size: {file_size:,} bytes ({file_size / (1024 * 1024):.2f} MiB)")

    with open(file_path, "rb") as f:
        first_bytes = f.read(2048)
        text_preview = first_bytes.decode("utf-8", errors="replace")[:500]

    print("First 500 characters preview:")
    print("--------------------------------------------------")
    print(text_preview)
    print("--------------------------------------------------")

    if count_newlines:
        newlines_count = 0
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                newlines_count += chunk.count(b"\n")
        print(f"Total newline count (\\n): {newlines_count:,}")


def main() -> None:
    """Download MOOCCubeX files and inspect the downloaded samples."""
    concept_problem_url = f"{BASE_URL}/relations/concept-problem.txt"
    concept_problem_path = RAW_DIR / "concept-problem.txt"

    user_problem_url = f"{BASE_URL}/relations/user-problem.json"
    user_problem_path = RAW_DIR / "user-problem-sample.jsonl"

    try:
        # 1. Download concept-problem.txt completely
        download_stream(
            url=concept_problem_url,
            output_path=concept_problem_path,
            limit_bytes=None,
        )

        # 2. Stream only the first 200 MiB of user-problem.json
        download_stream(
            url=user_problem_url,
            output_path=user_problem_path,
            limit_bytes=TARGET_SAMPLE_BYTES,
        )

        # 3. Inspect results
        inspect_file(concept_problem_path, count_newlines=False)
        inspect_file(user_problem_path, count_newlines=True)

    except Exception as exc:
        print(f"[FATAL] Downloader error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
