PALACE — a wordless 5-minute cinematic short film
=================================================
Directed / written / produced procedurally with Blender's Python API.

FILES
-----
  film_main.py      the complete production script (detect, build, render, assemble)
  film_project.blend the saved Blender project (resumable)
  final_film.mp4    the finished film, exactly 300.0s, 24 fps
  preview.mp4       low-resolution derivative of the finished cut
  audio.wav         the complete synthesized soundtrack (22.05 kHz mono)
  storyboard.txt    creative development + full shot-by-shot breakdown
  render_log.txt    environment report, benchmark, render times, assembly log
  frames/           rendered PNG frames (unique frames; safe to re-render)

REQUIREMENTS
------------
  * Blender 4.5 LTS-class Python module (pip package `bpy`) or Blender itself.
  * If bpy is not installed system-wide, run ./restore_env.sh once. It builds
    the in-repo environment under _env/ (gitignored): bpy 4.5.15 + numpy +
    imageio-ffmpeg wheels from PyPI, plus the headless X11/GL stub libraries
    bpy's loader needs on a GL-less machine. Then run ./blenderpy.sh film_main.py --all
  * ffmpeg (here: the static build bundled with the `imageio-ffmpeg` wheel).
  * No GPU required. No internet. No external assets. Fonts ship with bpy.

RUNNING
-------
  With the bpy module (as produced here):
      ./blenderpy.sh film_main.py --all

  With a real Blender installation:
      blender --background --python film_main.py -- --all

  Individual phases:
      --detect     environment/compatibility report (Phase 5)
      --build-only construct the scene and save film_project.blend
      --benchmark  render 8 representative frames, estimate total, auto-pick preset
      --render     render all frames (resume-safe; existing valid frames are skipped)
      --audio      synthesize audio.wav
      --assemble   build final_film.mp4 + preview.mp4

QUALITY PRESETS (auto-selected by benchmark; force with --preset A|B|C)
-----------------------------------------------------------------------
  A  640x360  8 fps grid (2400 unique frames)   6 samples
  B  480x270  8 fps grid (2400 unique frames)   6 samples
  C  426x240  8 fps grid (2400 unique frames)   4 samples
  All presets render 2400 unique frames on the fixed 8 fps animation grid so
  timing, shots and the exact 300.0 s runtime are identical at every preset.

THE CONFORMED 24 FPS (honest note)
-----------------------------------
final_film.mp4 plays at 24 fps. Because unique footage was rendered on an
8 fps grid, each rendered frame is DUPLICATED x3, not interpolated and not
truly animated at 24 fps. The lower cadence reads as deliberate stop-motion /
limited animation — the house style of the film.
Audio is cut to the same 300.0 s timeline either way.

CRASH RECOVERY
--------------
Re-run --render at any time. Frames that exist and pass PNG signature/size
checks are kept; only missing or corrupt frames are re-rendered. The .blend
is re-saved incrementally every 25 frames. Do NOT delete frames/ to retry —
it simply resumes.

MANUAL ASSEMBLY (if ffmpeg ever fails)
--------------------------------------
  ffmpeg -framerate 8 -i frames/frame_%04d.png -i audio.wav \
         -vf "fps=24,scale=640:360,fade=t=out:st=296:d=4" \
         -c:v libx264 -crf 20 -pix_fmt yuv420p -c:a aac -t 300 final_film.mp4
Preview: ffmpeg -i final_film.mp4 -vf scale=426:-2 -c:v libx264 -crf 28 preview.mp4

NOTES ON CRAFT
--------------
  * Exactly 300.0 seconds, 15 shots, one location, two characters.
  * No background music; the soundtrack is 100% synthesized (sea, rain,
    projector clatter, footsteps, thunder, one clock chime, one final click).
  * The boy's yellow raincoat and the white paper boat are the only
    saturated elements; everything else is desaturated teal/sepia.
  * Storyboard continuity note: the original treatment carried a 1953
    flashback in S12; the final cut replaces it with two wordless ECUs
    (the lit attic hatch; the boat on F7's armrest) — quieter, and cheaper
    to light. storyboard.txt reflects this change.
