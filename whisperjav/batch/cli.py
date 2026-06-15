from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .discovery import classify_video, discover_media_files
from .models import BatchOptions, ClassifiedVideo
from .nfo import extract_actresses_from_nfo, find_nfo_for_video
from .reports import BatchReportWriter, ReportPaths, summarize_results
from .scheduler import BatchScheduler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch transcribe and translate JAV videos.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--translate-api-key")
    parser.add_argument("--translate-workers", type=_translate_workers, default=1)
    parser.add_argument("--translation-queue-size", type=_translation_queue_size, default=2)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--include-audio", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-translate", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--accept-cpu-mode", action="store_true")
    parser.add_argument("--no-nfo", action="store_true")
    parser.add_argument("--actress")

    args = parser.parse_args(argv)
    if args.force and args.force_translate:
        parser.error("--force and --force-translate are mutually exclusive")
    if not args.root.exists() or not args.root.is_dir():
        parser.error("root must be an existing directory")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    options = BatchOptions(
        root=args.root,
        report_dir=args.report_dir,
        include_audio=args.include_audio,
        dry_run=args.dry_run,
        force=args.force,
        force_translate=args.force_translate,
        translate_api_key=args.translate_api_key,
        translate_workers=args.translate_workers,
        translation_queue_size=args.translation_queue_size,
        stream=args.stream,
        debug=args.debug,
        accept_cpu_mode=args.accept_cpu_mode,
        no_nfo=args.no_nfo,
        actress=args.actress,
    )
    report_dir = options.report_dir or options.root / ".whisperjav_batch"
    media_files = discover_media_files(options.root, include_audio=options.include_audio)
    classified = [_attach_actress_context(classify_video(path, options), options) for path in media_files]
    results = BatchScheduler(options).run(classified)
    paths = BatchReportWriter(report_dir=report_dir, input_root=options.root).write(results)
    summary = summarize_results(results)
    _print_summary(summary, paths)
    return 1 if summary["failed"] else 0


def _translate_workers(value: str) -> int:
    count = _positive_int(value, "--translate-workers")
    if count > 4:
        raise argparse.ArgumentTypeError("--translate-workers must be between 1 and 4")
    return count


def _translation_queue_size(value: str) -> int:
    return _positive_int(value, "--translation-queue-size")


def _positive_int(value: str, option: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{option} must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"{option} must be at least 1")
    return parsed


def _attach_actress_context(item: ClassifiedVideo, options: BatchOptions) -> ClassifiedVideo:
    manual_actresses = _manual_actresses(options.actress)
    if manual_actresses:
        return replace(item, nfo_path=None, actresses=manual_actresses)
    if options.no_nfo:
        return item

    lookup = find_nfo_for_video(item.video_path)
    warnings = list(item.warnings)
    if lookup.reason:
        warnings.append(lookup.reason)
    if lookup.path is None:
        return replace(item, warnings=tuple(warnings))

    parsed = extract_actresses_from_nfo(lookup.path)
    if parsed.error:
        warnings.append(parsed.error)
    return replace(
        item,
        nfo_path=lookup.path,
        actresses=parsed.actresses,
        warnings=tuple(warnings),
    )


def _manual_actresses(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _print_summary(summary: dict, paths: ReportPaths) -> None:
    print("WHISPERJAV BATCH SUMMARY")
    print(f"total_videos: {summary['total_videos']}")
    for status, count in summary["counts_by_status"].items():
        print(f"{status}: {count}")
    print(f"Report: {paths.jsonl}")
    print(f"Summary: {paths.summary}")
