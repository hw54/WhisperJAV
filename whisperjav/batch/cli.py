from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

from .discovery import classify_video, discover_media_files
from .models import BatchOptions, ClassifiedVideo, VideoResult
from .nfo import extract_metadata_from_nfo, find_nfo_for_video
from .progress import BatchProgressReporter, TqdmBatchProgress
from .reports import BatchReportWriter, ReportPaths, summarize_results
from .scheduler import BatchScheduler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch transcribe and translate JAV videos.")
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--translate-api-key")
    parser.add_argument("--translate-workers", type=_translate_workers, default=1)
    parser.add_argument("--translation-queue-size", type=_translation_queue_size, default=2)
    parser.add_argument("--asr-retries", type=_retry_count, default=1)
    parser.add_argument("--translation-retries", type=_retry_count, default=2)
    parser.add_argument("--max-video-minutes", type=_nonnegative_minutes, default=230)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--include-audio", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-translate", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--accept-cpu-mode", action="store_true")
    parser.add_argument("--no-nfo", action="store_true")
    parser.add_argument("--actress")

    args = parser.parse_args(argv)
    if args.force and args.force_translate:
        parser.error("--force and --force-translate are mutually exclusive")
    for root in args.roots:
        if not root.exists() or not root.is_dir():
            parser.error(f"root must be an existing directory: {root}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summaries = _run_roots(args)
    return 1 if any(summary["failed"] for summary in summaries) else 0


def _run_root(args: argparse.Namespace, root: Path) -> dict:
    return _run_roots(args, roots=[root])[0]


def _run_roots(args: argparse.Namespace, roots: list[Path] | None = None) -> list[dict]:
    selected_roots = list(roots or args.roots)
    root_items: list[tuple[Path, ClassifiedVideo]] = []

    for root in selected_roots:
        options = _build_options(args, root, multiple_roots=len(selected_roots) > 1)
        media_files = discover_media_files(options.root, include_audio=options.include_audio)
        for path in media_files:
            root_items.append((root, _attach_nfo_context(classify_video(path, options), options)))

    scheduler_options = _build_options(args, selected_roots[0], multiple_roots=len(selected_roots) > 1)
    progress = _create_progress_reporter(args)
    started = time.perf_counter()
    results = BatchScheduler(scheduler_options, progress=progress).run(item for _, item in root_items)
    wall_seconds = time.perf_counter() - started
    results_by_root = _split_results_by_root(selected_roots, root_items, results)
    summaries = []

    for root in selected_roots:
        root_results = results_by_root[root]
        report_dir = _report_dir_for_root(args, root, multiple_roots=len(selected_roots) > 1)
        root_wall_seconds = wall_seconds if len(selected_roots) == 1 else None
        paths = BatchReportWriter(report_dir=report_dir, input_root=root).write(
            root_results,
            wall_seconds=root_wall_seconds,
        )
        summary = summarize_results(root_results, wall_seconds=root_wall_seconds)
        _print_summary(summary, paths)
        summaries.append(summary)

    if len(selected_roots) > 1:
        _print_aggregate_summary(summarize_results(results, wall_seconds=wall_seconds))

    return summaries


def _build_options(args: argparse.Namespace, root: Path, *, multiple_roots: bool = False) -> BatchOptions:
    options = BatchOptions(
        root=root,
        report_dir=_report_dir_for_root(args, root, multiple_roots=multiple_roots),
        include_audio=args.include_audio,
        dry_run=args.dry_run,
        force=args.force,
        force_translate=args.force_translate,
        translate_api_key=args.translate_api_key,
        translate_workers=args.translate_workers,
        translation_queue_size=args.translation_queue_size,
        asr_retries=args.asr_retries,
        translation_retries=args.translation_retries,
        max_video_minutes=args.max_video_minutes,
        stream=args.stream,
        no_progress=args.no_progress,
        debug=args.debug,
        accept_cpu_mode=args.accept_cpu_mode,
        no_nfo=args.no_nfo,
        actress=args.actress,
    )
    return options


def _split_results_by_root(
    roots: list[Path],
    root_items: list[tuple[Path, ClassifiedVideo]],
    results: list[VideoResult],
) -> dict[Path, list[VideoResult]]:
    results_by_root = {root: [] for root in roots}
    for (root, _), result in zip(root_items, results, strict=True):
        results_by_root[root].append(result)
    return results_by_root


def _report_dir_for_root(args: argparse.Namespace, root: Path, *, multiple_roots: bool = False) -> Path:
    if args.report_dir is None:
        return root / ".whisperjav_batch"
    if not multiple_roots:
        return args.report_dir
    return args.report_dir / _root_report_subdir(root)


def _root_report_subdir(root: Path) -> str:
    digest = hashlib.sha1(str(root.resolve()).encode("utf-8")).hexdigest()[:8]
    safe_name = re.sub(r"[^\w.-]+", "_", root.name).strip("._")
    return f"{safe_name or 'root'}-{digest}"


def _translate_workers(value: str) -> int:
    return _positive_int(value, "--translate-workers")


def _translation_queue_size(value: str) -> int:
    return _positive_int(value, "--translation-queue-size")


def _retry_count(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("retry count must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("retry count must be at least 0")
    return parsed


def _nonnegative_minutes(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--max-video-minutes must be a number") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("--max-video-minutes must be at least 0")
    return parsed


def _positive_int(value: str, option: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{option} must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"{option} must be at least 1")
    return parsed


def _attach_nfo_context(item: ClassifiedVideo, options: BatchOptions) -> ClassifiedVideo:
    manual_actresses = _manual_actresses(options.actress)
    if options.no_nfo:
        if manual_actresses:
            return replace(item, actresses=manual_actresses)
        return item

    lookup = find_nfo_for_video(item.video_path)
    warnings = list(item.warnings)
    if lookup.reason:
        warnings.append(lookup.reason)
    if lookup.path is None:
        if manual_actresses:
            return replace(item, actresses=manual_actresses, warnings=tuple(warnings))
        return replace(item, warnings=tuple(warnings))

    parsed = extract_metadata_from_nfo(lookup.path)
    if parsed.error:
        warnings.append(parsed.error)
    return replace(
        item,
        nfo_path=lookup.path,
        actresses=manual_actresses or parsed.actresses,
        movie_title=parsed.movie_title,
        movie_plot=parsed.movie_plot,
        warnings=tuple(warnings),
    )


def _manual_actresses(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _create_progress_reporter(args: argparse.Namespace) -> BatchProgressReporter | None:
    if args.no_progress or args.dry_run:
        return None
    if args.stream:
        print("Batch progress disabled because --stream is enabled.", file=sys.stderr)
        return None
    return TqdmBatchProgress()


def _print_summary(summary: dict, paths: ReportPaths) -> None:
    print("WHISPERJAV BATCH SUMMARY")
    print(f"total_videos: {summary['total_videos']}")
    for status, count in summary["counts_by_status"].items():
        print(f"{status}: {count}")
    _print_performance(summary)
    print(f"Report: {paths.jsonl}")
    print(f"Summary: {paths.summary}")


def _print_aggregate_summary(summary: dict) -> None:
    print("WHISPERJAV BATCH AGGREGATE")
    print(f"total_videos: {summary['total_videos']}")
    for status, count in summary["counts_by_status"].items():
        print(f"{status}: {count}")
    _print_performance(summary)


def _print_performance(summary: dict) -> None:
    performance = summary.get("performance") or {}
    print(f"total_asr_seconds: {_format_seconds(summary.get('total_asr_seconds'))}")
    print(f"total_translation_seconds: {_format_seconds(summary.get('total_translation_seconds'))}")
    print(f"average_asr_seconds: {_format_seconds(performance.get('average_asr_seconds'))}")
    print(f"average_translation_seconds: {_format_seconds(performance.get('average_translation_seconds'))}")
    print(
        "average_total_processing_seconds: "
        f"{_format_seconds(performance.get('average_total_processing_seconds'))}"
    )
    print(f"total_video_duration_seconds: {_format_seconds(performance.get('total_video_duration_seconds'))}")
    print(f"total_wall_seconds: {_format_seconds(performance.get('total_wall_seconds'))}")
    print(
        "processing_to_duration_ratio: "
        f"{_format_ratio(performance.get('processing_to_duration_ratio'))}"
    )
    print(f"wall_to_duration_ratio: {_format_ratio(performance.get('wall_to_duration_ratio'))}")
    print(f"throughput_ratio: {_format_ratio(performance.get('throughput_ratio'))}")
    print(f"unknown_duration_count: {performance.get('unknown_duration_count', 0)}")
    print(f"duration_excluded_count: {performance.get('duration_excluded_count', 0)}")


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"
