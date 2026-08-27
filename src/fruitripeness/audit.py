"""Read-only ZIP audit. Never extracts, renames, or deletes source images."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import warnings
import zipfile

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

FRUITS = {"apple", "banana", "mango", "orange", "tomato"}
STAGES = {"ripe": "ripe", "unripe": "unripe", "overipe": "overripe", "overripe": "overripe"}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_labels(name: str) -> dict:
    """Accept an optional outer folder; preserve the original path and label typo."""
    parts = PurePosixPath(name).parts
    if len(parts) < 3:
        raise ValueError("Expected split/stage/image path")
    split, raw_stage = parts[-3].lower(), parts[-2].lower()
    tokens = PurePosixPath(name).stem.lower().split("_")
    if split not in {"train", "test", "validation", "val"}:
        raise ValueError(f"Unknown split: {split}")
    if raw_stage not in STAGES or tokens[0] not in FRUITS:
        raise ValueError("Unknown fruit or stage")
    stage = STAGES[raw_stage]
    if len(tokens) < 2 or STAGES.get(tokens[1]) != stage:
        raise ValueError("Filename and folder labels disagree")
    return {"source_split": split, "fruit": tokens[0], "stage": stage, "raw_stage": raw_stage}


def perceptual_hash(image: Image.Image) -> int:
    """64-bit DCT hash: candidate similarity only, not proof of duplication."""
    pixels = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    x = np.arange(32)
    k = np.arange(8)[:, None]
    basis = np.cos(np.pi * (2 * x + 1) * k / 64)
    dct = basis @ pixels @ basis.T
    values = dct.flatten()
    bits = values > np.median(values[1:])
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def duplicate_groups(rows: list[dict], key: str) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if row.get(key):
            groups[row[key]].append(row)
    result = []
    for group in groups.values():
        if len(group) > 1:
            result.append({
                "paths": [r["path"] for r in group],
                "cross_split": len({r["source_split"] for r in group}) > 1,
                "conflicting_labels": len({(r["fruit"], r["stage"]) for r in group}) > 1,
            })
    return result


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit_zip(source: Path, output: Path, distance: int = 4) -> dict:
    if not 0 <= distance <= 8:
        raise ValueError("Perceptual distance must be between 0 and 8")
    source, output = Path(source), Path(output)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset ZIP not found: {source}")
    output.mkdir(parents=True, exist_ok=True)
    rows, skipped, hashes = [], [], []
    with zipfile.ZipFile(source) as archive:
        entries = sorted((i for i in archive.infolist() if not i.is_dir()), key=lambda x: x.filename)
        for i, entry in enumerate(entries, 1):
            name = entry.filename
            if PurePosixPath(name).suffix.lower() not in EXTENSIONS:
                skipped.append(name)
                continue
            row = {"path": name, "bytes": entry.file_size, "error": ""}
            try:
                row.update(parse_labels(name))
                if entry.file_size > 50 * 1024 * 1024:
                    raise ValueError("Image exceeds 50 MiB audit limit")
                blob = archive.read(entry)
                row["sha256"] = hashlib.sha256(blob).hexdigest()
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(io.BytesIO(blob)) as original:
                        original.load()
                        rgb = ImageOps.exif_transpose(original).convert("RGB")
                        row.update(width=rgb.width, height=rgb.height, mode=original.mode, format=original.format)
                        payload = f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes()
                        row["pixel_sha256"] = hashlib.sha256(payload).hexdigest()
                        h = perceptual_hash(rgb)
                        row["phash"] = f"{h:016x}"
                        hashes.append((h, row))
            except (ValueError, OSError, UnidentifiedImageError, zipfile.BadZipFile,
                    Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            if i % 500 == 0:
                print(f"Decoded {i}/{len(entries)} files", flush=True)
    valid = [r for r in rows if not r["error"]]
    byte_groups = duplicate_groups(valid, "sha256")
    pixel_groups = duplicate_groups(valid, "pixel_sha256")
    # Exhaustive Hamming-distance search, practical for this ~4,600-image dataset.
    # Candidate pairs remain for human review and are NEVER automatically deleted.
    near = []
    for i, (ha, a) in enumerate(hashes):
        for hb, b in hashes[:i]:
            if a["pixel_sha256"] == b["pixel_sha256"]:
                continue
            d = (ha ^ hb).bit_count()
            if d <= distance:
                near.append({"path_a": b["path"], "path_b": a["path"], "distance": d,
                             "cross_split": a["source_split"] != b["source_split"],
                             "conflicting_labels": (a["fruit"], a["stage"]) != (b["fruit"], b["stage"])})
    counts = Counter((r["source_split"], r["fruit"], r["stage"]) for r in valid)
    summary = {
        "source_filename": source.name, "image_files": len(rows), "valid_images": len(valid),
        "invalid_images": len(rows)-len(valid), "skipped_files": skipped,
        "counts": [{"split": s, "fruit": f, "stage": t, "count": n} for (s,f,t),n in sorted(counts.items())],
        "dimensions": [{"width": w, "height": h, "count": n} for (w,h),n in Counter((r["width"],r["height"]) for r in valid).most_common()],
        "exact_byte_groups": len(byte_groups),
        "exact_pixel_groups": len(pixel_groups),
        "exact_pixel_redundant_copies": sum(len(g["paths"])-1 for g in pixel_groups),
        "exact_pixel_cross_split_groups": sum(g["cross_split"] for g in pixel_groups),
        "exact_pixel_conflicting_label_groups": sum(g["conflicting_labels"] for g in pixel_groups),
        "near_duplicate_threshold": distance, "near_duplicate_candidate_pairs": len(near),
        "near_duplicate_cross_split_pairs": sum(r["cross_split"] for r in near),
        "near_duplicate_conflicting_label_pairs": sum(r["conflicting_labels"] for r in near),
        "limitations": "Perceptual similarity is a review flag, not proof. No semantic label verification, physical-fruit identity inference, or leakage-free split is claimed."
    }
    fields = ["path","source_split","fruit","stage","raw_stage","bytes","width","height","mode","format","sha256","pixel_sha256","phash","error"]
    write_csv(output / "manifest.csv", rows, fields)
    write_csv(output / "near_duplicate_candidates.csv", near, ["path_a","path_b","distance","cross_split","conflicting_labels"])
    (output / "duplicate_groups.json").write_text(json.dumps({"byte_groups": byte_groups, "pixel_groups": pixel_groups}, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=Path("data/raw/archive.zip"))
    parser.add_argument("--out", type=Path, default=Path("outputs/audit"))
    parser.add_argument("--distance", type=int, default=4)
    args = parser.parse_args()
    result = audit_zip(args.zip, args.out, args.distance)
    print(json.dumps({k:v for k,v in result.items() if k not in {"counts","dimensions"}}, indent=2))
    if result["invalid_images"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
