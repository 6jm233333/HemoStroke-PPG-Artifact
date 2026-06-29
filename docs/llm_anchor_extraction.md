# Historical LLM Anchor Extraction Disclosure

## Preserved Historical Record

The original temporal-anchor extraction run used the Google model identifier
`gemini-3-pro-preview` between **2025-12-22** and **2025-12-29**. According to
the official Gemini API changelog, the model was released on 2025-11-18 and the
preview endpoint was discontinued on 2026-03-09. Google redirected the retired
model identifier to `gemini-3.1-pro-preview`.

Official lifecycle source:
<https://ai.google.dev/gemini-api/docs/changelog>

## Preserved Components

- Version-controlled prompt: `prompts/stroke_timestamp_extraction.md`
- Canonical input fields: `Row_ID`, `CHARTTIME`, `TEXT`
- Output fields: `Row_ID`, `Extracted_Timestamp`
- Null token: `NULL`
- Local post-processing code for MIMIC-III and MC-MED

The task is structured information extraction, not diagnosis. The prompt first
checks whether a current-encounter acute stroke-related event is documented. It
then prioritizes explicit onset, discovery, witnessed, or last-known-well
timestamps. If an eligible event is documented without a specific event time,
the record timestamp is used as a surrogate. Otherwise, the output is `NULL`.

## Parameters Not Preserved

The original request wrapper was not retained. The following request-level
controls cannot be retrospectively verified:

| Parameter | Historical value |
|---|---|
| Invocation transport | `not_preserved` |
| Region | `not_preserved` |
| `temperature` | `not_preserved` |
| `top_p` | `not_preserved` |
| `top_k` | `not_preserved` |
| `max_output_tokens` | `not_preserved` |
| Seed | `not_preserved` |
| Thinking configuration | `not_preserved` |

These values are deliberately not reconstructed from present-day API defaults.
Because the preview endpoint has been retired, an authorized rerun with a
successor model is a transparent reconstruction rather than a bitwise replay.

## Restricted-Data Boundary

Clinical source text, generated payloads, returned model outputs, reviewed
anchors, and derived patient-level indices are restricted local artifacts.
They must remain outside version control. Only the prompt, schemas, configs,
and local processing logic are released.
