# TUI Showroom

Scripted demos of `ui-tui`. Workflows snapshot real ui-tui components (`MessageLine`, `Panel`, `Box`, `Text`) into ANSI and replay them in the browser with cinematic overlays. Recorded once, played any number of times — built for screen capture.

```bash
npm run showroom            # dev server at http://127.0.0.1:4317
npm run showroom:record     # regenerate every workflow JSON
npm run showroom:build      # dist/<name>.html for every workflow
npm run showroom:type-check
```

## Bundled workflows

| File                            | Shows                                                          |
| ------------------------------- | -------------------------------------------------------------- |
| `workflows/feature-tour.json`   | Plan → tool trail → result highlight                           |
| `workflows/subagent-trail.json` | Parallel subagents, hot lanes, summary                         |
| `workflows/slash-commands.json` | `/skills`, `/model`, `/agents`, `/help` typed → echoed → panel |
| `workflows/voice-mode.json`     | VAD capture, transcript, TTS ducking                           |

Pick a workflow from the dropdown or deep-link with `?w=<name>`.

## Architecture

```
record.tsx           ─┐
  ↳ MessageLine,     │  Ink renders → Writable → ANSI string
    Panel, Box, Text │
                     ▼
workflows/<name>.json
                     │  served at /api/workflow/<name>
                     ▼
showroom.js          │  ANSI parser + DOM overlays targeting frame ids
                     ▼
browser
```

`frame` actions embed ANSI from an Ink render; the browser parses them into `<pre>` elements with a lightweight converter. Captions, spotlights, highlights, and fades are DOM overlays anchored to frame `id`s. No CDN dependencies — zero network latency.

## Timeline actions

| Action      | Required         | Optional                                      |
| ----------- | ---------------- | --------------------------------------------- |
| `frame`     | `ansi`           | `id`                                          |
| `status`    | `text`           | `detail`                                      |
| `compose`   | `text`           | `duration` (typewriter)                       |
| `caption`   | `target`, `text` | `position` (`left`/`right`/`top`), `duration` |
| `spotlight` | `target`         | `pad`, `duration`                             |
| `highlight` | `target`         | `duration`                                    |
| `fade`      | `target`         | `to` (default `0`), `duration`                |
| `clear`     | —                | —                                             |

`target` references the `id` of an earlier `frame`. `viewport.scale` (or the 1x–4x picker) controls the upscale factor for capture.

## Player

- Restart (`R`), 1x–4x scale, 0.5x/1x/2x speed (`1`/`2`/`3`).
- Progress bar reads `at + duration` from the slowest action.

## Adding a workflow

1. Add a scene fn to `record.tsx` returning `{ title, viewport, composer, timeline }`.
2. Compose Ink primitives or pull `MessageLine` / `Panel` from `../src`.
3. `await snap(<Component />)` for each frame.
4. `npm run showroom:record`.

Components must be state-free at first paint — `useEffect` hooks won't fire by the time the recorder unmounts. For accordions like the live `ToolTrail`, render a flat `Box` + `Text` scene instead.
