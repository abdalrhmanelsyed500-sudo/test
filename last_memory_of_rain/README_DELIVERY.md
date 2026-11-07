# THE LAST MEMORY OF RAIN — Delivery README

A short surreal CG film about a lone keeper (M-07) on a desiccated station who fires a
seeding rocket into a storm cell at the edge of a dead desert — and waits for the rain.

Silent picture with an agent-synthesized soundtrack. Titles fade in/out in the final shot.

## Watch

- `THE_LAST_MEMORY_OF_RAIN_FINAL.mp4` — 1920×810, 24 fps, H.264 + AAC (≈2:45). This is the deliverable.
- `THE_LAST_MEMORY_OF_RAIN_preview.mp4` (if present) — 1280×540 smaller preview copy.
- `audio_master_165s.wav` — full stereo audio master (165.0 s, 44.1 kHz PCM), as used in the MP4.

## Technical specs

| Property | Value |
|---|---|
| Runtime | 165.0 s exactly |
| Animation grid | 640×270 @ 8 fps, 1320 frames (stop-motion cadence) |
| Delivery format | Upscaled ×3 nearest to 1920×810 @ 24 fps (frame triplication) |
| Codec | H.264 (CRF 17), yuv420p; audio AAC |
| Engine | Blender 4.5.15 (bpy module), Cycles CPU, PRESET A (6 spp @ 640×270) |
| Look | AgX, exposure −0.9 (world 0.75); Bayer 4×4 36-level dither, split-tone teal/ember, S-curve, halftone on lit areas, CA, per-frame grain, vignette |
| Color | Amber/copper dusk sky, deep teal shadows; halftone-dither print grade |

## Pipeline

1. `source/film_main.py --render` — resumable Cycles render of 1320 frames to `frames/`.
2. `source/film_main.py --post` — print grade (Bayer dither / split-tone / halftone / CA / grain / vignette).
3. `source/film_main.py --assemble` — triplicate frames, H.264 + AAC mux, preview copy.
4. `source/audio_synth.py` — regenerates `audio.wav` (pure numpy synthesis; no external samples).
5. `source/restore_env.sh` — rebuilds the in-repo Python environment (bpy wheel, numpy, imageio-ffmpeg, stub libs).

## File inventory

| File | Purpose |
|---|---|
| `THE_LAST_MEMORY_OF_RAIN_FINAL.mp4` | Deliverable (1920×810, 24 fps, H.264+AAC) |
| `audio_master_165s.wav` | Audio master used in the film |
| `storyboard.md` | Shot table, frame map, delivery specs |
| `render_log.txt` | Full build/render/post/assemble log |
| `source/film_main.py` | Blender/Cycles scene, camera, animation, render, post, assembly |
| `source/audio_synth.py` | Soundtrack synthesizer |
| `source/blenderpy.sh`, `source/restore_env.sh` | Environment wrapper / environment rebuild |

## Provenance & disclosures

- **The screenplay/story/shot design for this film was written by an AI agent** as part of an
  agentic production task; it is an original agent invention, not an adaptation.
- Visuals: procedural geometry and procedural canvas textures created in code; AI-generated imagery
  was used for none of the film's frames (all frames are path-traced renders of the coded scene).
- Audio: 100 % agent-synthesized (numpy); no licensed or third-party samples.
- No third-party IP was used; no likenesses; no logos.
- M-07, the station, the annex, the terminal (RAIN-9 "61%"), the seeding rocket, and the sprout are fictional.

## Contact

Produced in an Arena.ai Agent Mode workspace; sources under `last_memory_of_rain/` on branch
`arena/792d0627-test` of the session repository.
