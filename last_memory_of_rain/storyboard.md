# THE LAST MEMORY OF RAIN — Shot Production Plan

**Provenance note:** No six-scene screenplay existed in the conversation when this film was ordered.
This screenplay was authored by the production agent from the user's stated beats (M-07 the abandoned
weather-station robot, the last seed, the sacrifice of its energy core, first rain after decades of
drought, six scenes). It is original and wordless by design.

**Runtime:** 165.0 s exactly. Animation grid 8 fps → 1320 unique frames, conformed to 24 fps by
frame triplication at assembly (stop-motion cadence, PALACE-style limited animation).
**Native res:** 640×270 (2.37:1) → upscaled ×3 nearest-neighbor to 1920×810. No 16:9 stretch, no bars.
**Style spec:** `reference/CINEMATIC_2.5D_PIXEL_ART_VISUAL_PRODUCTION_BIBLE.md` (kept in repo reference/).
**Audio:** stereo 44.1 kHz, original Karplus-Strong-style score + original sound design. No dialogue.
Red is reserved for M-07's scarf. Cream is reserved for emitted light. Teal/orange environmental axis.

## Six-scene screenplay (wordless)

**S01 — EXT. RIDGE — DAWN (12 s).** A dead weather annex on a ridge above a petrified valley.
Dawn. M-07 — a small keeper robot with a red scarf — walks its morning perimeter toward the annex
and enters. (The ritual older than drought.)

**S02 — INT. ANNEX — THE LEDGER (24 s).** The rain gauge: **0.0 mm**. Below it:
**11,324 consecutive days.** M-07 opens the drawer: in amber glass, the **last seed**. It holds it
up to the light the way it was never taught to.

**S03 — EXT. STATION — THE ANOMALY (18 s).** On the horizon: a **dark cell** under a copper sky.
A flock crosses. Inside, the long-silent forecast terminal cranks to life by itself: **61 % — RAIN.**
The first nonzero reading in two decades. The wind sock lifts, once. M-07 goes still.

**S04 — INT. ANNEX — THE ARITHMETIC (21 s).** Two cores on the workbench: the **seeding rocket's
launch core** (built decades ago for drought relief, never fired) and **M-07's own keeper cell** —
the cell that keeps a little robot alive. One full charge, two purposes. The calculation runs in
light around the cores and resolves. The answer has been obvious for a very long time.

**S05 — EXT. LAUNCH PAD — THE CHOICE (20 s).** M-07 carries the rocket core out across the pad.
Joints gutter (lens dimming). It mounts the core. The station dies panel by panel behind it.
The rocket climbs. Heat lightning answers from inside the cell.

**S06 — EXT. THE FIRST RAIN — DAWN (~57 s).** Night. Heat lightning. And then — **one macro drop**
strikes a single blade of grass. Downpour. Dawn comes up washed and silver. On the pad, in the
seed glass: **a green sprout.** M-07's lens clicks back on. Title: **THE LAST MEMORY OF RAIN.**

## Shot table (frame ranges @ 8 fps; Σ = 1320 = 165.0 s)

| # | Shot | Frames | Len | Description |
|---|------|--------|-----|-------------|
| 01 | S01-A | 1–72 | 8.0 s | EXT wide: approach across the crust; monument scale vs tiny figure |
| 02 | S01-B | 73–128 | 7.5 s | EXT: low pass beneath the dome; M-07 enters the annex |
| 03 | S02-A | 129–240 | 14.0 s | INT: slow push toward M-07 at the cradle — the gauge ledger wall |
| 04 | S02-B | 241–336 | 12.0 s | INT: the drawer; amber glass; the last seed held to the light |
| 05 | S03-A | 337–392 | 7.0 s | EXT: the dark cell on the horizon; birds cross |
| 06 | S03-B | 457–536(393–456) | 8.0 s | INT: terminal self-starts; **61 % RAIN** resolves; rain sock lifts |
| 07 | S04-A | 537–624 | 11.0 s | INT: workbench ECU — rocket core vs keeper cell; holo arcs spin |
| 08 | S04-B | 625–704 | 10.0 s | INT: the choice — M-07 reaches, takes the rocket core |
| 09 | S05-A | 705–800 | 12.0 s | EXT: the pad; the rocket core lowered into the cradle; station dies |
| 10 | S05-B | 801–880 | 10.0 s | EXT: ignition — climb; heat lightning |
| 11 | S06-A | 881–968 | 9.0 s | MACRO: one raindrop strikes a grass blade; the ring spreads |
| 12 | S06-B | 969–1076 | 13.5 s | EXT wide: downpour over the valley; M-07 slumped at the pad |
| 13 | S06-C | 1077–1180 | 13.0 s | EXT: dawn; the seed glass — a green sprout |
| 14 | S06-D | 1181–1320 | 17.5 s | EXT: M-07 rises; lens clicks; title card: THE LAST MEMORY OF RAIN |

(Frame ranges as implemented in `film_main.py`.)

## Camera grammar
- Deep focus, locked/drifting frames; cuts only on shot boundaries.
- 2.37:1 compositions: horizon low, sky dominant, annex as mass, M-07 as scale-marker.
- No handheld shake; the only "handheld" is the launch rumble (rocket wobble curve).

## Color grammar
- Teal (shadow/air) ↔ ember orange (sun/fire) master axis; cream = emitted sources only
  (lamps, cores, lens, terminal, rain glyph, title card).
- Red = M-07's scarf ONLY. Never reused elsewhere, never for warnings/text.
- Storm = desaturated teal-slate; dawn S06 = silvered warm-grey with green sprout accent.

## Print grammar (in-pipeline, not a flat filter)
1. Split-tone grade (shadows→teal, highlights→ember), gentle S-curve.
2. Bayer 4×4 ordered dither, 36-level quantization.
3. Coarse halftone dots in illuminated regions.
4. Radial chromatic aberration (≤1.4 px at corners).
5. Animated film grain (luminance-aware, per-frame seeded).
6. Vignette.
Composite chain: Cycles → numpy grade pass (`--post`) → nearest-neighbor upscale at assembly.

## Sound plan (stereo 44.1 kHz)
Wind/low drone → lonely piano motif (S02) → low pulse tension (S03–S04) → launch rumble (S05) →
first drop = high piano tone + thunder (S06) → rain bed → resolved motif under title. No dialogue.

## Delivery
- `THE_LAST_MEMORY_OF_RAIN_FINAL.mp4` — 1920×810, 24 fps (8 fps grid ×3), H.264 High@4.1, AAC 160k.
- Google Drive folder **"THE LAST MEMORY OF RAIN — FINAL DELIVERY"** with MP4, preview, audio, sources, README.
