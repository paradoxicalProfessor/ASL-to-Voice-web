import argparse
import os
import re
from pathlib import Path


def gather_existing_max_indices(dataset_root: Path):
    # scan dataset/*/labels for files like A_0001.txt and return max index per prefix
    pattern = re.compile(r"^([A-Z])_(\d{4})\.txt$")
    max_idx = {}
    for subset in ('train', 'valid', 'test'):
        labels_dir = dataset_root / subset / 'labels'
        if not labels_dir.exists():
            continue
        for p in labels_dir.iterdir():
            if not p.is_file():
                continue
            m = pattern.match(p.name)
            if m:
                prefix, num = m.group(1), int(m.group(2))
                max_idx[prefix] = max(max_idx.get(prefix, 0), num)
    return max_idx


def find_images(dirpath: Path):
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    return [p for p in sorted(dirpath.iterdir()) if p.is_file() and p.suffix.lower() in exts]


def infer_prefix_from_name(fname: str):
    # find first ascii letter a-z or A-Z in filename and return uppercase
    for ch in fname:
        if ch.isalpha():
            return ch.upper()
    return 'X'


def main():
    p = argparse.ArgumentParser(description='Rename images in "custom dataset" to dataset PREFIX_#### format')
    p.add_argument('--custom-dir', default='custom dataset', help='Folder with images to rename')
    p.add_argument('--dataset-root', default='dataset', help='Root of existing dataset to read current indices')
    p.add_argument('--dry-run', action='store_true', help='Show mappings without applying')
    args = p.parse_args()

    custom = Path(args.custom_dir)
    if not custom.exists() or not custom.is_dir():
        print('Custom dataset folder not found:', custom)
        return

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print('Dataset root not found, proceeding with start indices = 0')

    max_idx = gather_existing_max_indices(dataset_root)

    imgs = find_images(custom)
    if not imgs:
        print('No images found in', custom)
        return

    # counters per prefix
    counters = {k: v + 1 for k, v in max_idx.items()}  # next available

    mappings = []
    for img in imgs:
        prefix = infer_prefix_from_name(img.name)
        if prefix not in counters:
            counters[prefix] = 1
        idx = counters[prefix]
        ext = img.suffix.lower()
        new_name = f"{prefix}_{idx:04d}{ext}"
        new_path = custom / new_name
        mappings.append((img, new_path, prefix, idx))
        counters[prefix] += 1

    # check collisions
    collisions = [new for (_, new, _, _) in mappings if new.exists() and new not in [old for (old, _, _, _) in mappings]]
    if collisions:
        print('Collision detected, aborting:')
        for c in collisions:
            print('  ', c)
        return

    for old, new, prefix, idx in mappings:
        print(f"{old} -> {new}")

    if args.dry_run:
        print('Dry run complete. No files changed.')
        return

    # apply two-phase renaming
    tmp_paths = []
    try:
        for old, new, _, _ in mappings:
            tmp = old.with_suffix(old.suffix + '.renametmp')
            os.rename(old, tmp)
            tmp_paths.append((tmp, new))
        for tmp, final in tmp_paths:
            os.rename(tmp, final)
    except Exception as e:
        print('Error during renaming:', e)
        # try rollback
        for tmp, final in tmp_paths:
            if tmp.exists():
                try:
                    os.rename(tmp, tmp.with_suffix(''))
                except Exception:
                    pass
        return

    print('Renaming complete.')


if __name__ == '__main__':
    main()
