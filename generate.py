#!/usr/bin/env python3
"""Generator dataset OCR — template-based (editează imaginea, nu o construiește de la zero)."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

from tqdm import tqdm

from ro_id_synth.debug_grid import build_debug_grid
from ro_id_synth.pipeline import write_debug_triplet
from ro_id_synth.records import generate_record
from ro_id_synth.template_config import load_template_spec, save_fields_overlay
from ro_id_synth.worker import _process_batch, init_pool


def merge_label_parts(labels_dir: Path, out_file: Path) -> None:
    parts = sorted(labels_dir.glob("part_*.txt"))
    with out_file.open("w", encoding="utf-8") as out:
        for part in parts:
            text = part.read_text(encoding="utf-8")
            if text:
                out.write(text if text.endswith("\n") else text + "\n")
            part.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generator template-based CI românesc")
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--output", type=str, default="dataset")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fields",
        type=str,
        default=str(Path(__file__).parent / "config" / "template_fields.json"),
        help="template_fields.json — coordonate câmpuri",
    )
    parser.add_argument("--debug-grid", action="store_true")
    parser.add_argument("--grid-count", type=int, default=100)
    parser.add_argument("--analyze", action="store_true", help="Salvează overlay câmpuri + triplet debug")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.resolve()
    fields_path = Path(args.fields).resolve()
    out_root = Path(args.output).resolve()
    debug_dir = out_root / "debug"

    spec = load_template_spec(fields_path, base_dir)
    if not spec.image_path.is_file():
        raise FileNotFoundError(f"Template lipsă: {spec.image_path}")

    if args.analyze:
        debug_dir = Path(args.output).resolve() / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        save_fields_overlay(spec, debug_dir / "fields_overlay.jpg")
        import numpy as np

        rng = np.random.default_rng(args.seed)
        write_debug_triplet(spec, generate_record(seed=args.seed), rng, debug_dir)
        print(f"Debug: {debug_dir}/fields_overlay.jpg, original.jpg, replaced_fields.jpg, final_sample.jpg")
        return

    if args.debug_grid:
        debug_dir.mkdir(parents=True, exist_ok=True)
        build_debug_grid(str(fields_path), str(base_dir), debug_dir / "grid_0001.jpg", count=args.grid_count, seed=args.seed)
        print(f"Grid: {debug_dir / 'grid_0001.jpg'} ({args.grid_count} sample-uri)")
        return

    images_dir = out_root / "images"
    labels_dir = out_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Triplet debug la start
    import numpy as np

    write_debug_triplet(spec, generate_record(seed=args.seed), np.random.default_rng(args.seed), debug_dir)

    batch_size = max(1, args.batch_size)
    total = args.count
    num_batches = (total + batch_size - 1) // batch_size
    worker_args = [
        (
            b,
            min(batch_size, total - b * batch_size),
            b * batch_size,
            str(images_dir),
            str(labels_dir),
            args.seed,
            str(debug_dir),
            str(fields_path),
            str(base_dir),
        )
        for b in range(num_batches)
    ]

    print(f"Template: {spec.image_path.name} | {total} imagini | {args.workers} workers")
    t0 = time.perf_counter()
    init_args = (str(fields_path), str(base_dir))

    if args.workers <= 1:
        init_pool(*init_args)
        for wa in tqdm(worker_args, desc="batch"):
            _process_batch(wa)
    else:
        with mp.Pool(processes=args.workers, initializer=init_pool, initargs=init_args) as pool:
            list(tqdm(pool.imap_unordered(_process_batch, worker_args), total=len(worker_args), desc="batch"))

    merge_label_parts(labels_dir, labels_dir / "train.txt")
    elapsed = time.perf_counter() - t0
    print(f"Gata: {images_dir} ({total} imagini, {elapsed:.1f}s)")
    print(f"Debug: {debug_dir}/")

    build_debug_grid(str(fields_path), str(base_dir), debug_dir / "grid_final.jpg", count=min(100, total), seed=args.seed)


if __name__ == "__main__":
    mp.freeze_support()
    main()
