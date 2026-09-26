#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["Pillow>=9.0"]
# ///
"""Sample image pairs and compare the Rust wgpu implementation to FLIP CPU/CUDA."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


WORKSPACE = Path(__file__).resolve().parents[1]
FLIP_ROOT = (WORKSPACE / "../../Github/flip").resolve()
DEFAULT_IMAGES = FLIP_ROOT / "images"
DEFAULT_ENGINES = ["wgpu-gpu", "wgpu-cpu", "flip-cpp", "flip-cuda"]
SCORE_KEYS = [
    "mean",
    "weighted_median",
    "first_weighted_quartile",
    "third_weighted_quartile",
    "min",
    "max",
]
SCORE_PATTERNS = {
    "mean": r"Mean:\s*([0-9.eE+-]+)",
    "weighted_median": r"Weighted median:\s*([0-9.eE+-]+)",
    "first_weighted_quartile": r"1st weighted quartile:\s*([0-9.eE+-]+)",
    "third_weighted_quartile": r"3rd weighted quartile:\s*([0-9.eE+-]+)",
    "min": r"Min:\s*([0-9.eE+-]+)",
    "max": r"Max:\s*([0-9.eE+-]+)",
    "evaluation_seconds": r"Evaluation time:\s*([0-9.eE+-]+)\s*seconds",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample image pairs and compare wgpu FLIP to FLIP's CPU/CUDA tools."
    )
    parser.add_argument("--manifest", type=Path, help="CSV with name,reference,test columns")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--reference", type=Path, help="Reference image for --tests")
    parser.add_argument("--tests", type=Path, nargs="+", help="Test images paired with --reference")
    parser.add_argument("--fraction", type=float, default=0.5, help="Random fraction of cases to run (default: 0.5)")
    parser.add_argument("--limit", type=int, help="Optional maximum number of sampled cases")
    parser.add_argument("--seed", type=int, default=20260925, help="Repeatable random sampling seed")
    parser.add_argument("--engines", nargs="+", choices=DEFAULT_ENGINES, default=DEFAULT_ENGINES)
    parser.add_argument("--output-dir", type=Path, help="Artifact directory (default: target/flip-bench-TIMESTAMP)")
    parser.add_argument("--score-tolerance", type=float, default=1e-5)
    parser.add_argument("--map-channel-tolerance", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-engine timeout in seconds")
    parser.add_argument("--no-build", action="store_true", help="Do not cargo build --release first")
    return parser.parse_args()


def discover_cases(args: argparse.Namespace) -> list[dict[str, Path | str]]:
    if args.manifest:
        manifest = args.manifest.resolve()
        with manifest.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows or not {"reference", "test"}.issubset(rows[0]):
            raise ValueError("manifest must contain CSV columns named `reference` and `test`")
        cases = []
        for index, row in enumerate(rows, 1):
            ref = Path(row["reference"])
            test = Path(row["test"])
            if not ref.is_absolute():
                ref = manifest.parent / ref
            if not test.is_absolute():
                test = manifest.parent / test
            cases.append({"name": row.get("name") or f"case-{index:04d}", "reference": ref.resolve(), "test": test.resolve()})
        return cases

    if args.reference or args.tests:
        if not args.reference or not args.tests:
            raise ValueError("provide both --reference and --tests")
        reference = args.reference.resolve()
        return [
            {"name": test.stem, "reference": reference, "test": test.resolve()}
            for test in args.tests
        ]

    image_dir = args.images_dir.resolve()
    cases = []
    for extension in ("png", "exr"):
        reference = image_dir / f"reference.{extension}"
        test = image_dir / f"test.{extension}"
        if reference.is_file() and test.is_file():
            cases.append({"name": f"bundled-{extension}", "reference": reference, "test": test})
    if not cases:
        raise ValueError(
            f"no bundled reference/test pairs found in {image_dir}; use --manifest for a larger corpus"
        )
    return cases


def select_cases(cases: list[dict[str, Path | str]], fraction: float, limit: int | None, seed: int):
    if not 0.0 < fraction <= 1.0:
        raise ValueError("--fraction must be in (0, 1]")
    count = max(1, math.ceil(len(cases) * fraction))
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        count = min(count, limit)
    rng = random.Random(seed)
    return sorted(rng.sample(cases, min(count, len(cases))), key=lambda item: str(item["name"]))


def safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return name or "case"


def build_command(engine: str, case: dict, output_map: Path) -> list[str]:
    reference = str(case["reference"])
    test = str(case["test"])
    hdr = Path(reference).suffix.lower() == ".exr"
    if engine.startswith("wgpu-"):
        executable = WORKSPACE / "target/release/wgpu_test"
        command = [
            str(executable), "--reference", reference, "--test", test,
            "--output", str(output_map), "--verbosity", "2", "--no-exposure-map",
        ]
        if engine == "wgpu-cpu":
            command.append("--force-fallback-adapter")
        return command

    executable = FLIP_ROOT / "build" / ("flip-cuda" if engine == "flip-cuda" else "flip")
    command = [
        str(executable), "--reference", reference, "--test", test,
        "--directory", str(output_map.parent), "--basename", output_map.stem, "--verbosity", "2",
    ]
    if hdr:
        command.append("--no-exposure-map")
    return command


def parse_result(output: str) -> dict:
    result = {}
    for name, pattern in SCORE_PATTERNS.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            result[name] = float(match.group(1))
    adapter = re.search(r"Adapter:\s*(.+)", output)
    if adapter:
        result["adapter"] = adapter.group(1).strip()
    return result


def compare_maps(left: Path, right: Path) -> dict:
    with Image.open(left) as left_file, Image.open(right) as right_file:
        image_a = left_file.convert("RGB")
        image_b = right_file.convert("RGB")
    if image_a.size != image_b.size:
        return {"same_size": False, "left_size": image_a.size, "right_size": image_b.size}
    pixels_a = image_a.tobytes()
    pixels_b = image_b.tobytes()
    exact_pixels = 0
    maximum_delta = 0
    sum_delta = 0
    channel_count = image_a.width * image_a.height * 3
    for offset in range(0, channel_count, 3):
        delta_r = abs(pixels_a[offset] - pixels_b[offset])
        delta_g = abs(pixels_a[offset + 1] - pixels_b[offset + 1])
        delta_b = abs(pixels_a[offset + 2] - pixels_b[offset + 2])
        maximum_delta = max(maximum_delta, delta_r, delta_g, delta_b)
        sum_delta += delta_r + delta_g + delta_b
        exact_pixels += delta_r == 0 and delta_g == 0 and delta_b == 0
    pixel_count = image_a.width * image_a.height
    return {
        "same_size": True,
        "width": image_a.width,
        "height": image_a.height,
        "exact_pixels": exact_pixels,
        "exact_pixel_fraction": exact_pixels / pixel_count,
        "max_channel_delta": maximum_delta,
        "mean_absolute_channel_delta": sum_delta / channel_count,
    }


def write_partial_report(path: Path, report: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, report: dict) -> None:
    rows = []
    for case in report["cases"]:
        for engine, result in case["engines"].items():
            row = {
                "case": case["name"],
                "reference": case["reference"],
                "test": case["test"],
                "engine": engine,
                "status": result["status"],
                "adapter": result.get("adapter", ""),
                "wall_seconds": result.get("wall_seconds", ""),
                "evaluation_seconds": result.get("evaluation_seconds", ""),
                "mean": result.get("mean", ""),
                "weighted_median": result.get("weighted_median", ""),
                "first_weighted_quartile": result.get("first_weighted_quartile", ""),
                "third_weighted_quartile": result.get("third_weighted_quartile", ""),
                "min": result.get("min", ""),
                "max": result.get("max", ""),
                "error": result.get("error", ""),
            }
            rows.append(row)
    fields = [
        "case", "reference", "test", "engine", "status", "adapter", "wall_seconds",
        "evaluation_seconds", *SCORE_KEYS, "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    try:
        cases = discover_cases(args)
        selected = select_cases(cases, args.fraction, args.limit, args.seed)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (WORKSPACE / "target" / f"flip-bench-{stamp}")
    output_dir = output_dir.resolve()
    maps_dir = output_dir / "maps"
    logs_dir = output_dir / "logs"
    maps_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    binary = WORKSPACE / "target/release/wgpu_test"
    if not args.no_build and any(engine.startswith("wgpu-") for engine in args.engines):
        build = subprocess.run(["cargo", "build", "--release"], cwd=WORKSPACE, text=True)
        if build.returncode:
            print("cargo build --release failed", file=sys.stderr)
            return build.returncode
    if any(engine.startswith("wgpu-") for engine in args.engines) and not binary.is_file():
        print(f"missing Rust executable: {binary}", file=sys.stderr)
        return 2

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(WORKSPACE),
        "flip_root": str(FLIP_ROOT),
        "fraction": args.fraction,
        "seed": args.seed,
        "engines": args.engines,
        "score_tolerance": args.score_tolerance,
        "map_channel_tolerance": args.map_channel_tolerance,
        "cases": [],
    }
    report_path = output_dir / "report.json"
    runs_path = output_dir / "runs.jsonl"
    runs_path.write_text("", encoding="utf-8")
    print(f"Selected {len(selected)} of {len(cases)} cases (seed {args.seed}); writing {output_dir}", flush=True)

    all_ok = True
    for case_index, raw_case in enumerate(selected, 1):
        name = safe_name(str(raw_case["name"]))
        case_report = {
            "name": name,
            "reference": str(raw_case["reference"]),
            "test": str(raw_case["test"]),
            "engines": {},
            "map_comparisons": {},
            "accuracy_ok": None,
        }
        map_paths = {}
        print(f"[{case_index}/{len(selected)}] {name}: {case_report['reference']} vs {case_report['test']}", flush=True)
        for engine in args.engines:
            output_map = maps_dir / f"{name}.{engine}.png"
            log_path = logs_dir / f"{name}.{engine}.log"
            command = build_command(engine, raw_case, output_map)
            started = time.perf_counter()
            try:
                process = subprocess.run(
                    command,
                    cwd=WORKSPACE if engine.startswith("wgpu-") else FLIP_ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                )
                wall_seconds = time.perf_counter() - started
                log_path.write_text(process.stdout, encoding="utf-8")
                parsed = parse_result(process.stdout)
                result = {
                    **parsed,
                    "status": "ok" if process.returncode == 0 else "failed",
                    "returncode": process.returncode,
                    "wall_seconds": wall_seconds,
                    "command": command,
                    "map": str(output_map) if output_map.is_file() else None,
                    "log": str(log_path),
                }
                if process.returncode != 0:
                    result["error"] = process.stdout[-4000:]
                    all_ok = False
                elif "mean" not in parsed or "evaluation_seconds" not in parsed:
                    result["status"] = "failed"
                    result["error"] = "could not parse Mean and Evaluation time from tool output"
                    all_ok = False
                elif not output_map.is_file():
                    result["status"] = "failed"
                    result["error"] = "engine succeeded but did not create its error map"
                    all_ok = False
                else:
                    map_paths[engine] = output_map
                print(
                    f"  {engine}: {result['status']}; evaluation {result.get('evaluation_seconds', 'n/a')}s; "
                    f"wall {wall_seconds:.3f}s; mean {result.get('mean', 'n/a')}",
                    flush=True,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                wall_seconds = time.perf_counter() - started
                log_path.write_text(str(error), encoding="utf-8")
                result = {
                    "status": "failed",
                    "error": str(error),
                    "wall_seconds": wall_seconds,
                    "command": command,
                    "log": str(log_path),
                }
                all_ok = False
                print(f"  {engine}: failed: {error}", flush=True)
            case_report["engines"][engine] = result
            with runs_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"case": name, "engine": engine, **result}, sort_keys=True) + "\n")

        oracle_engines = [engine for engine in ("flip-cpp", "flip-cuda") if engine in map_paths]
        candidate_engines = [engine for engine in ("wgpu-gpu", "wgpu-cpu") if engine in map_paths]
        accuracy_ok = None
        if oracle_engines and candidate_engines:
            accuracy_ok = True
            for candidate in candidate_engines:
                for oracle in oracle_engines:
                    comparison = compare_maps(map_paths[candidate], map_paths[oracle])
                    key = f"{candidate}_vs_{oracle}"
                    case_report["map_comparisons"][key] = comparison
                    candidate_result = case_report["engines"][candidate]
                    oracle_result = case_report["engines"][oracle]
                    score_deltas = {
                        score: abs(candidate_result[score] - oracle_result[score])
                        for score in SCORE_KEYS
                        if score in candidate_result and score in oracle_result
                    }
                    case_report.setdefault("score_deltas", {})[key] = score_deltas
                    scores_ok = bool(score_deltas) and max(score_deltas.values()) <= args.score_tolerance
                    maps_ok = comparison.get("same_size", False) and comparison["max_channel_delta"] <= args.map_channel_tolerance
                    accuracy_ok = accuracy_ok and scores_ok and maps_ok
        case_report["accuracy_ok"] = accuracy_ok
        if accuracy_ok is False:
            all_ok = False
        accuracy_label = "PASS" if accuracy_ok else ("FAIL" if accuracy_ok is False else "SKIP (no candidate/oracle pair)")
        print(f"  map/score match within configured tolerances: {accuracy_label}", flush=True)
        report["cases"].append(case_report)
        write_partial_report(report_path, report)
        write_csv(output_dir / "results.csv", report)

    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["overall_ok"] = all_ok
    write_partial_report(report_path, report)
    write_csv(output_dir / "results.csv", report)
    print(f"Report: {report_path}")
    print(f"Run log: {runs_path}")
    print(f"CSV: {output_dir / 'results.csv'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
