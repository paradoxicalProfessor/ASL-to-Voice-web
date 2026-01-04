#!/usr/bin/env python3
"""Preview or apply bulk renaming of image/label pairs in a YOLO dataset.

Usage:
  python scripts/rename_dataset.py --dataset-root dataset --subset test --dry-run

Behavior:
  - Reads dataset/data.yaml to map class ids to names.
  - For each image in dataset/<subset>/images, determines corresponding label in dataset/<subset>/labels.
  - Creates a new basename like `<CLASS>_0001.jpg` when the label contains at least one object and first class is available; otherwise `image_0001.jpg`.
  - With `--dry-run` prints proposed renames; without it performs the filesystem changes.
"""
import argparse
import sys
import os
from pathlib import Path

import yaml


def load_names(dataset_root: Path):
    cfg = dataset_root / "data.yaml"
    if not cfg.exists():
        return None
    with cfg.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("names")


def find_images(images_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts and p.is_file()])


def read_first_class(label_path: Path):
    if not label_path.exists():
        return None
    try:
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 1:
                    return parts[0]
    except Exception:
        return None
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="dataset", help="Path to dataset folder")
    p.add_argument("--subset", default="all", help="Which subset: train, valid, test, or all")
    p.add_argument("--start-index", type=int, default=1, help="Start index for numbering")
    p.add_argument("--dry-run", action="store_true", help="Only print proposed changes")
    args = p.parse_args()

    root = Path(args.dataset_root)
    if not root.exists():
        print("Dataset root not found:", root)
        sys.exit(1)

    names = load_names(root)
    if names:
        names_map = {str(i): names[i] for i in range(len(names))}
    else:
        names_map = {}

    subsets = [args.subset] if args.subset != "all" else ["train", "valid", "test"]
    mappings = []

    for subset in subsets:
        images_dir = root / subset / "images"
        labels_dir = root / subset / "labels"
        if not images_dir.exists():
            print(f"Skipping missing subset: {subset}")
            continue
        labels_dir.mkdir(parents=True, exist_ok=True)

        imgs = find_images(images_dir)
        counter = args.start_index
        for img in imgs:
            base = img.stem
            label = labels_dir / (base + ".txt")
            first_cls = read_first_class(label)
            if first_cls is not None and first_cls in names_map:
                prefix = names_map[first_cls]
            elif first_cls is not None:
                prefix = f"class{first_cls}"
            else:
                prefix = "image"

            new_base = f"{prefix}_{counter:04d}"
            new_img = images_dir / (new_base + img.suffix.lower())
            new_label = labels_dir / (new_base + ".txt")
            mappings.append((img, new_img, label, new_label))
            counter += 1

    if not mappings:
        print("No files found to rename.")
        return

    dests = {str(new_img) for (_, new_img, _, _) in mappings}
    if len(dests) != len(mappings):
        print("Name collisions detected in proposed targets. Aborting.")
        sys.exit(1)

    for old_img, new_img, old_label, new_label in mappings:
        rel_old = os.path.relpath(old_img, Path.cwd())
        rel_new = os.path.relpath(new_img, Path.cwd())
        print(f"{rel_old} -> {rel_new}")
        if old_label.exists():
            rel_old_label = os.path.relpath(old_label, Path.cwd())
            rel_new_label = os.path.relpath(new_label, Path.cwd())
            print(f"  label: {rel_old_label} -> {rel_new_label}")

    if args.dry_run:
        print("Dry run complete. No files changed.")
        return

    for old_img, new_img, old_label, new_label in mappings:
        if new_img.exists():
            print(f"Target exists, skipping: {new_img}")
            continue
        old_img.rename(new_img)
        if old_label.exists():
            old_label.rename(new_label)

    print("Renaming complete.")


if __name__ == "__main__":
    main()
