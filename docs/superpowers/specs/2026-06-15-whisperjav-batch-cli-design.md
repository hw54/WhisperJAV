# WhisperJAV Batch CLI Design

Date: 2026-06-15

## Goal

Add a formal CLI entry point, `whisperjav-batch`, for the user's recurring JAV library workflow:

1. Accept one root directory.
2. Recursively find videos under that directory.
3. Skip videos that already have matching external subtitles.
4. Reuse existing WhisperJAV Japanese subtitles when present.
5. Generate missing Japanese subtitles with the recommended Anime-Whisper configuration.
6. Translate missing Chinese subtitles with DeepSeek in Adult/Explicit style.
7. Run ASR and translation in an overlapped but stable pipeline.
8. Write a terminal summary and durable JSONL/JSON reports.

The CLI is intended to replace repeated GUI setup for long batch runs while keeping failures visible and resumable.

## Non-Goals

- Do not add a GUI flow.
- Do not implement a persistent ASR worker in the first version.
- Do not run multiple ASR jobs concurrently.
- Do not use an LLM to parse NFO files in the first version.
- Do not translate third-party external subtitles in the first version; a matching non-WhisperJAV subtitle remains an external subtitle blocker.
- Do not hide subprocess failures or silently downgrade configured behavior.
- Do not rework `whisperjav.main` batch semantics beyond what this wrapper needs.

## Entry Point

Add a package script:

```toml
whisperjav-batch = "whisperjav.batch.cli:main"
```

Basic usage:

```bash
whisperjav-batch /path/to/root
```

The command scans recursively and writes reports under:

```text
<input_root>/.whisperjav_batch/
```

## Default ASR Configuration

For videos requiring transcription, the wrapper invokes the existing CLI as a subprocess with one input file at a time:

```bash
python -u -m whisperjav.main <video> \
  --ensemble \
  --ensemble-serial \
  --pass1-pipeline qwen \
  --pass1-sensitivity balanced \
  --pass1-qwen-params '{"framer":"vad-grouped","generator_backend":"anime-whisper","timestamp_mode":"vad_only","assembly_cleaner":"passthrough","stepdown":false}' \
  --pass1-scene-detector semantic \
  --pass1-speech-segmenter whisperseg \
  --pass1-model litagin/anime-whisper \
  --merge-strategy pass1_primary \
  --output-dir source \
  --subs-language native \
  --language japanese
```

`--ensemble-serial` is harmless for single-file invocations and makes the command intent explicit.

Implementation must build the subprocess with `sys.executable`, not a hardcoded `python`, and force UTF-8/unbuffered IO for stable log streaming:

```text
[sys.executable, "-u", "-m", "whisperjav.main", ...]
```

The wrapper does not pass Pass 2 options by default.

## Default Translation Configuration

The wrapper invokes translation as a separate subprocess through the existing translation CLI. This keeps cloud/API translation failures isolated from the batch controller and matches the ASR subprocess boundary.

Default translation subprocess:

```bash
python -u -m whisperjav.translate.cli \
  -i <japanese_srt> \
  --provider deepseek \
  --source japanese \
  --target chinese \
  --tone pornify \
  --model deepseek-v4-flash
```

Implementation must build this with `sys.executable`:

```text
[sys.executable, "-u", "-m", "whisperjav.translate.cli", ...]
```

Do not use `whisperjav.main --translate-*` names for this subprocess. Those names belong to the main transcription CLI; the standalone translation CLI uses `--provider`, `--target`, `--tone`, `--model`, and `--actress`. The standalone CLI supports argv API keys, but the batch wrapper must not put secrets in subprocess argv.

API key resolution:

1. Batch CLI `--translate-api-key`, if provided, is injected into the translation subprocess environment as `DEEPSEEK_API_KEY`.
2. Otherwise, provider-specific environment variables are resolved by the translation CLI/service.

The existing translation settings file can provide provider/model/tone-style preferences to `whisperjav.translate.cli`, but it does not store API keys. Because this batch workflow has fixed recommended defaults, the wrapper passes provider/model/target/tone explicitly by default.

When NFO or manual metadata produces actress context, pass it as:

```text
--actress "Name1, Name2"
```

The expected translation output path follows the same naming rule as `whisperjav.translate.cli.generate_output_path(input, "chinese")`: split the SRT stem on `.`, and if the final dot-separated segment is a recognized language code (`japanese`, `english`, `ja`, `en`, `jp`), remove that final segment before appending `.chinese.srt`. For the recommended ASR output this yields:

```text
ABC-123.ja.pass1.srt -> ABC-123.ja.pass1.chinese.srt
```

The ASR subprocess does not receive `--translate`; translation is scheduled by the wrapper after a Japanese SRT exists.

## File Discovery

Video extensions come from `MediaDiscovery.video_extensions`:

```text
.mp4, .avi, .mkv, .mov, .wmv, .flv, .webm, .m4v, .mpg, .mpeg
```

Pure audio files are not included by default. `--include-audio` may opt into `MediaDiscovery.audio_extensions`.

The scan is recursive. Duplicate paths are resolved by canonical path.

## Subtitle Classification

For each video `ABC-123.mp4`, classify by basename and same directory.

External subtitle extensions:

```text
.srt, .vtt, .ass, .ssa, .sub
```

External subtitles match the video basename, including language suffixes:

```text
ABC-123.srt
ABC-123.zh.srt
ABC-123.ass
```

These external subtitles are blockers in v1, even if they appear to be source-language subtitles rather than Chinese subtitles. The first version only reuses WhisperJAV-generated Japanese subtitles.

WhisperJAV-generated subtitles are not treated as external subtitles:

```text
ABC-123.ja.pass1.srt
ABC-123.ja.whisperjav.srt
ABC-123.ja.merged.whisperjav.srt
ABC-123.ja.pass1.chinese.srt
ABC-123.ja.whisperjav.chinese.srt
ABC-123.ja.merged.whisperjav.chinese.srt
```

Chinese translation detection checks:

1. The expected translation path derived from the selected Japanese SRT:
   `(<japanese_srt>.stem).chinese.srt`.
2. Compatible WhisperJAV Chinese outputs:
   `ABC-123*.chinese.srt`, `ABC-123*.zh.srt`, `ABC-123*.cn.srt`, limited to WhisperJAV-style stems where possible.

Existing translated SRTs are reusable only after validation:

- file is readable as UTF-8 with replacement for invalid bytes;
- stripped content is non-empty;
- at least two subtitle blocks contain a ` --> ` timecode marker, matching the main pipeline's minimum translation precheck.

If a candidate translation exists but fails validation, record `invalid_existing_translation` in the reason. If a reusable Japanese WhisperJAV SRT exists, schedule translation again. If not, process the video according to normal classification unless an external subtitle blocks it.

Classification statuses:

- `skip_external_subtitle`: matching external subtitle exists.
- `skip_translated`: Chinese translation already exists.
- `translate_existing_japanese`: Japanese WhisperJAV SRT exists, Chinese translation missing.
- `transcribe_then_translate`: no external subtitle and no reusable translated output.
- `failed_asr`: ASR failed.
- `failed_translation`: translation failed.
- `failed_precondition`: requested mode cannot run because a required reusable input is missing.
- `cancelled`: task did not reach a terminal success/failure state because the batch run was interrupted.
- `completed`: ASR and translation completed during this run.
- `completed_translation_only`: existing Japanese SRT translated during this run.

## NFO Actress Extraction

The wrapper looks for NFO metadata before scheduling translation.

Lookup order:

1. Same basename: `ABC-123.nfo`.
2. If no basename match exists and the directory has exactly one `.nfo`, use that file.
3. If multiple unmatched NFO files exist, do not guess; record `nfo_ambiguous`.

Parsing is deterministic and local. The first version does not call an LLM.

Supported fields:

- XML local names are matched case-insensitively.
- `<actor><name>...</name></actor>` has priority over direct actor text.
- direct `<actor>...</actor>` text is used only when the actor element has no child `<name>`.
- `<actress>...</actress>`, `<cast>...</cast>`, and `<performer>...</performer>` are supported.
- Non-XML NFO parsing is out of scope for v1.

Field values are split on comma, semicolon, Japanese/Chinese comma variants, and newlines. Names are normalized by trimming whitespace, removing line breaks, collapsing repeated spaces, preserving order, and deduplicating exact matches.

Multiple actresses are joined with `", "`, matching the GUI placeholder and the existing raw prompt path:

```text
--actress "Name1, Name2"
```

The existing code passes the value through as:

```text
Actress: Name1, Name2
```

If parsing fails, the wrapper records the parse error and continues without actress context.

Manual override:

```bash
--actress "Name1, Name2"
```

This has priority over NFO extraction. `--no-nfo` disables NFO parsing.

## Scheduling Model

First version uses stable overlapped scheduling:

```text
scan
  -> ASR video 1
      -> enqueue translation video 1
  -> ASR video 2 while translation video 1 runs
      -> enqueue translation video 2
  -> ...
wait for translation queue
write summary
```

ASR:

- Exactly one ASR subprocess at a time.
- One video per subprocess.
- Keeps ROCm/MIGraphX/HIP failures isolated.
- Avoids concurrent GPU model contention.

Translation:

- Background worker pool.
- Each translation task launches one translation subprocess.
- Default `--translate-workers 1`.
- Allowed range: `1..4`; invalid values are explicit errors.
- Default bounded queue size: `--translation-queue-size 2`.
- Queue insertion happens only after the current ASR subprocess has completed. If the queue is full, the scheduler blocks before launching the next ASR subprocess.
- Translation workers receive immutable task records and return immutable results.
- A single coordinator owns final per-video status updates and report writes.

Measured on the current ROCm machine, repeated Anime-Whisper + WhisperSeg model ready cost is roughly 10-12 seconds per subprocess. For 1-2 hour videos this is small relative to ASR and translation time, so stability is preferred over a persistent ASR worker in the first version.

## Error Handling

Failure policy:

- ASR failure records `failed_asr`, stores redacted command details, return code, and captured tail logs, then continues.
- Translation failure records `failed_translation`, stores redacted command details and error details, then continues.
- If any ASR or translation task fails, final exit code is `1`.
- If all actionable files succeed or are skipped, final exit code is `0`.
- Ctrl+C stops launching new ASR work, terminates the current ASR subprocess, terminates running translation subprocesses, marks queued/not-started translation tasks as `cancelled`, writes the partial report, and exits non-zero.

The wrapper must not synthesize success outputs, swallow subprocess failures, or silently downgrade configured behavior.

Subprocess shutdown is explicit: terminate first, then kill after a short fixed timeout. The timeout is part of signal cleanup only; it is not a hidden cap on normal ASR or translation execution.

## Reports

Default report directory:

```text
<input_root>/.whisperjav_batch/
```

Files:

```text
run-YYYYMMDD-HHMMSS.jsonl
run-YYYYMMDD-HHMMSS.summary.json
```

Each JSONL row represents one video:

```json
{
  "video_path": "...",
  "status": "completed",
  "reason": null,
  "nfo_path": "...",
  "actresses": ["Name1", "Name2"],
  "japanese_srt": "...",
  "chinese_srt": "...",
  "asr": {
    "seconds": 123.4,
    "return_code": 0,
    "command_redacted": ["...", "--pass1-model", "litagin/anime-whisper"]
  },
  "translation": {
    "seconds": 56.7,
    "return_code": 0,
    "api_key_source": "env",
    "command_redacted": ["...", "--model", "deepseek-v4-flash"]
  },
  "error": null,
  "warnings": []
}
```

Reports must never contain raw API keys. Translation API keys are passed through `DEEPSEEK_API_KEY` in the subprocess environment, not command argv. Redact values after `--api-key` and `--translate-api-key` if either secret flag is ever present in a command record, even when `--debug` is enabled.

The summary JSON includes:

- input root
- start/end timestamps
- total videos discovered
- counts by status
- total ASR seconds
- total translation seconds
- failure list
- report file path

Terminal summary prints the same status counts and report paths.

## CLI Options

Recommended first-version options:

```text
root
--dry-run
--translate-api-key
--translate-workers
--translation-queue-size
--report-dir
--include-audio
--force
--force-translate
--stream
--debug
--accept-cpu-mode
--no-nfo
--actress
```

Behavior:

- `--dry-run`: scan, classify, print/write reports, execute nothing.
- `--force`: ignore existing WhisperJAV Japanese and Chinese outputs and rerun ASR + translation unless an external subtitle blocks the video.
- `--force-translate`: do not rerun ASR; retranslate existing Japanese WhisperJAV SRTs. If no reusable Japanese WhisperJAV SRT exists, record `failed_precondition`.
- `--force` and `--force-translate` are mutually exclusive.
- `--stream`: stream subprocess output to terminal while also retaining enough log context for reports.
- `--debug`: pass debug flags to underlying commands and preserve full non-secret command context in reports.
- `--accept-cpu-mode`: pass through to ASR subprocess.

## Testing Strategy

Unit tests:

- discovery deduplicates recursive video files.
- external subtitle detection only uses same-basename subtitles.
- WhisperJAV Japanese subtitles are reusable, not external blockers.
- Chinese translation detection handles `.chinese.srt`, `.zh.srt`, and `.cn.srt`.
- NFO basename match wins over directory-level unique NFO.
- multiple unmatched NFO files produce `nfo_ambiguous`.
- NFO actor/name priority, localname case-insensitivity, delimiter splitting, and ambiguous-NFO error recording.
- ASR command contains the expected Anime-Whisper/WhisperSeg arguments and no Pass 2.
- translation command contains the standalone translation CLI arguments accepted by `argparse`: `--provider`, `--target`, `--tone`, `--model`, and `--actress`.
- generated reports redact API keys in all command fields, including debug mode.

Scheduler tests use fake runners:

- ASR success enqueues translation.
- ASR failure does not enqueue translation.
- translation failure does not stop later translations.
- bounded translation queue applies backpressure.
- queue backpressure blocks before launching the next ASR subprocess.
- Ctrl+C writes a partial report, terminates subprocesses, and marks pending translation tasks as `cancelled`.
- final exit code is non-zero when any actionable task fails.
- dry-run writes classification reports without invoking runners.

Verification commands for implementation should include focused tests for the new package plus any touched shared helpers, with the backend unit test timeout capped at 60 seconds.
