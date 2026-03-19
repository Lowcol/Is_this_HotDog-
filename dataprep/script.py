from __future__ import annotations

import argparse
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


SOURCE_CLASSES = ["hotdog", "people", "pets", "furniture"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create train/test folders with a fixed 25/25/25/25 source mix:\n"
            "hotdog, people, pets, furniture. Output is binary:\n"
            "hotdog (only hotdog files) and not_hotdog (people/pets/furniture)."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("dataset/hotdog/train_kaggle"),
        help="Source directory containing images to classify by filename.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/data"),
        help="Destination root directory for train/test folders.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned operations without writing files.",
    )
    return parser.parse_args()


def detect_class(path: Path, input_dir: Path) -> str | None:
    # Prefer folder labels relative to input_dir, so parent path names
    # like ".../dataset/hotdog/..." do not force a wrong class.
    rel_parts = [part.lower() for part in path.relative_to(input_dir).parts[:-1]]
    for class_name in SOURCE_CLASSES:
        if class_name in rel_parts:
            return class_name

    # Accept common singular/plural folder variants for non-hotdog classes.
    for token in rel_parts:
        if token in {"person", "people"}:
            return "people"
        if token in {"pet", "pets"}:
            return "pets"
        if token in {"furniture", "furnitures"}:
            return "furniture"

    # Fallback to filename prefix matching.
    name = path.stem.lower()
    if name.startswith("hotdog_"):
        return "hotdog"
    if name.startswith("people_") or name.startswith("person_"):
        return "people"
    if name.startswith("pets_") or name.startswith("pet_"):
        return "pets"
    if name.startswith("furniture_"):
        return "furniture"

    # Final fallback: token presence in filename.
    tokens = re.split(r"[^a-z0-9]+", name)
    for class_name in SOURCE_CLASSES:
        if class_name in tokens:
            return class_name
    if "person" in tokens:
        return "people"
    if "pet" in tokens:
        return "pets"
    return None


def collect_images(input_dir: Path) -> Tuple[Dict[str, List[Path]], int]:
    images_by_class: Dict[str, List[Path]] = {class_name: [] for class_name in SOURCE_CLASSES}
    total_image_files = 0

    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        total_image_files += 1

        class_name = detect_class(path, input_dir)
        if class_name is None:
            continue
        images_by_class[class_name].append(path)

    return images_by_class, total_image_files


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path

    stem, suffix = path.stem, path.suffix
    index = 1
    while True:
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def copy_or_move_file(src: Path, dst: Path, move: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    rng = random.Random(args.seed)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}\n"
            "Make sure your images are extracted there, or pass --input-dir."
        )

    images_by_class, total_image_files = collect_images(input_dir)
    counts = {class_name: len(paths) for class_name, paths in images_by_class.items()}

    if any(count == 0 for count in counts.values()):
        missing = [name for name, count in counts.items() if count == 0]
        raise RuntimeError(
            f"No images found for classes: {', '.join(missing)}.\n"
            "Check filenames contain: hotdog, people, pets, furniture."
        )

    balanced_per_class = min(counts.values())
    train_per_class = int(balanced_per_class * 0.8)
    test_per_class = balanced_per_class - train_per_class

    print(f"Input dir: {input_dir.resolve()}")
    print(f"Detected image files: {total_image_files}")
    print("Found images per source class:")
    for class_name in SOURCE_CLASSES:
        print(f"  - {class_name}: {counts[class_name]}")
    print()
    print(f"Using balanced subset: {balanced_per_class} images per source class")
    print(
        "Per split composition: "
        "25% hotdog, 25% people, 25% pets, 25% furniture"
    )
    print(
        f"Per source class split: train={train_per_class}, test={test_per_class}"
    )
    print(
        "Output mapping: hotdog -> hotdog folder, "
        "people/pets/furniture -> not_hotdog folder"
    )
    print(f"Output root: {output_dir}")
    print(f"Mode: {'move' if args.move else 'copy'}")
    print(f"Dry run: {args.dry_run}")
    print()

    operations = []
    for class_name in SOURCE_CLASSES:
        candidates = list(images_by_class[class_name])
        rng.shuffle(candidates)
        selected = candidates[:balanced_per_class]

        train_items = selected[:train_per_class]
        test_items = selected[train_per_class:]
        target_class = "hotdog" if class_name == "hotdog" else "not_hotdog"

        for src in train_items:
            dst = unique_destination(output_dir / "train" / target_class / src.name)
            operations.append((src, dst))

        for src in test_items:
            dst = unique_destination(output_dir / "test" / target_class / src.name)
            operations.append((src, dst))

    if args.dry_run:
        for src, dst in operations:
            print(f"[DRY RUN] {src} -> {dst}")
        print(f"\nPlanned operations: {len(operations)}")
        return

    for src, dst in operations:
        copy_or_move_file(src, dst, move=args.move)

    print(f"Done. Processed {len(operations)} images.")
    print(f"Train folder: {output_dir / 'train'}")
    print(f"Test folder: {output_dir / 'test'}")


if __name__ == "__main__":
    main()
