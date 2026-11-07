#!/bin/bash
# Rebuild the PALACE build environment inside this repo (self-contained).
# Usage: bash _env/restore_env.sh
# Recreates: pylibs/ (bpy 4.5.15 + numpy 1.26.4 + imageio-ffmpeg) and
#            stublibs/ (headless X11/GL stubs that bpy's RTLD_NOW loader needs).
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYLIBS="$HERE/pylibs"; STUBS="$HERE/stublibs"; WHEELS="$HERE/wheels"
mkdir -p "$PYLIBS" "$STUBS" "$WHEELS"

echo "== resolve wheel URLs =="
python3 - <<'PY'
import json, sys, urllib.request
want = {"bpy": "4.5.15", "numpy": "1.26.4", "imageio-ffmpeg": "0.5.1"}
cptag = "cp%d%d" % (sys.version_info.major, sys.version_info.minor)
outs = []
for pkg, ver in want.items():
    d = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/{ver}/json", timeout=60))
    url = None
    for f in d["urls"]:                       # prefer manylinux x86_64 wheels for THIS interpreter
        n = f["filename"]
        if n.endswith(".whl") and cptag in n and "manylinux" in n and "x86_64" in n:
            url = f["url"]; break
    if url is None:
        for f in d["urls"]:
            n = f["filename"]
            if n.endswith(".whl") and "manylinux" in n and "x86_64" in n:
                url = f["url"]; break
    if url is None:
        for f in d["urls"]:
            if f["filename"].endswith(".whl"):
                url = f["url"]; break
    outs.append((url, url.rsplit("/", 1)[1]))
with open("/tmp/wheel_urls.txt", "w") as fh:
    for url, fn in outs:
        fh.write(f"{url} {fn}\n")
PY

echo "== download wheels (resumable) =="
while read -r url fn; do
  [ -f "$WHEELS/$fn.done" ] && { echo "have $fn"; continue; }
  for i in $(seq 1 30); do
    echo "download $fn (attempt $i)"
    if curl -sS --retry 3 --max-time 1500 -C - "$url" -o "$WHEELS/$fn"; then
      touch "$WHEELS/$fn.done"; break
    fi
    sleep 2
  done
done < /tmp/wheel_urls.txt

echo "== unpack wheels =="
PALACE_ENV="$HERE" python3 - <<'PY'
import glob, os, shutil, zipfile
import os
HERE = os.environ.get("PALACE_ENV", "/home/user/test/palace_film/_env")
WHEELS = os.path.join(HERE, "wheels"); PYLIBS = os.path.join(HERE, "pylibs")
for stale in ("imageio_ffmpeg",):  # drop any stale macOS unpack
    d = os.path.join(PYLIBS, stale)
    if os.path.isdir(d): shutil.rmtree(d)
for whl in glob.glob(os.path.join(WHEELS, "*.whl")):
    print("unpack", os.path.basename(whl), flush=True)
    with zipfile.ZipFile(whl) as z:
        z.extractall(PYLIBS)
# ffmpeg binary must be executable (wheel loses the bit on some unpackers)
for root, _, files in os.walk(os.path.join(PYLIBS, "imageio_ffmpeg")):
    for f in files:
        if f.startswith("ffmpeg"):
            os.chmod(os.path.join(root, f), 0o755)
PY

echo "== rebuild stub libraries =="
cd "$STUBS"
cat > stub_empty.c <<'EOF'
int __stub_placeholder__(void) { return 0; }
EOF
cat > stub_xkb.c <<'EOF'
/* libxkbcommon stub: bpy's loader needs the V_0.5.0 version node + these 21
   symbols; never called headless. */
void *xkb_compose_state_feed(void) { return 0; }
void *xkb_compose_state_get_status(void) { return 0; }
void *xkb_compose_state_get_utf8(void) { return 0; }
void *xkb_compose_state_new(void) { return 0; }
void *xkb_compose_state_reset(void) { return 0; }
void *xkb_compose_state_unref(void) { return 0; }
void *xkb_compose_table_new_from_locale(void) { return 0; }
void *xkb_compose_table_unref(void) { return 0; }
void *xkb_context_new(void) { return 0; }
void *xkb_context_unref(void) { return 0; }
void *xkb_keymap_key_repeats(void) { return 0; }
void *xkb_keymap_mod_get_index(void) { return 0; }
void *xkb_keymap_new_from_string(void) { return 0; }
void *xkb_keymap_unref(void) { return 0; }
void *xkb_state_get_keymap(void) { return 0; }
void *xkb_state_key_get_one_sym(void) { return 0; }
void *xkb_state_key_get_utf8(void) { return 0; }
void *xkb_state_new(void) { return 0; }
void *xkb_state_serialize_mods(void) { return 0; }
void *xkb_state_unref(void) { return 0; }
void *xkb_state_update_mask(void) { return 0; }
EOF
cat > stub_gl.c <<'EOF'
/* Minimal libGL / libOpenGL stub: the 16 symbols bpy's libs reference under
   RTLD_NOW. Never called in a headless CPU-only render. */
void glFinish(void) {}
void *glXChooseFBConfig(void) { return 0; }
void *glXCreateContext(void) { return 0; }
void *glXCreateNewContext(void) { return 0; }
void *glXCreateWindow(void) { return 0; }
void glXDestroyContext(void) {}
void *glXGetCurrentContext(void) { return 0; }
void *glXGetCurrentDisplay(void) { return 0; }
void *glXGetCurrentDrawable(void) { return 0; }
void *glXGetProcAddress(void) { return 0; }
void *glXGetProcAddressARB(void) { return 0; }
void *glXGetVisualFromFBConfig(void) { return 0; }
void glXMakeContextCurrent(void) {}
void glXMakeCurrent(void) {}
void glXQueryContext(void) {}
void glXSwapBuffers(void) {}
EOF
cat > stub_xext.c <<'EOF'
/* X11 extension stubs for headless bpy import (never called). */
void XFixesHideCursor(void) {}
void XFixesShowCursor(void) {}
void XCloseDevice(void) {}
void XFreeDeviceList(void) {}
void XFreeDeviceState(void) {}
void XGetExtensionVersion(void) {}
void XListInputDevices(void) {}
void XOpenDevice(void) {}
void XQueryDeviceState(void) {}
void XSelectExtensionEvent(void) {}
void _XiGetDevicePresenceNotifyEvent(void) {}
void XtCreatePopupShell(void) {}
void XtDestroyWidget(void) {}
void *XtDisplay(void) { return 0; }
void XtOpenApplication(void) {}
void XtRealizeWidget(void) {}
void XtUnrealizeWidget(void) {}
void *XtWindow(void) { return 0; }
long XtStrings;
long topLevelShellWidgetClass;
long applicationShellWidgetClass;
EOF
cat > zstdtrace.c <<'EOF'
/* LD_PRELOAD: zstd trace hooks bpy probes (never called headless). */
void ZSTD_trace_compress_begin(void) {}
void ZSTD_trace_compress_end(void) {}
void ZSTD_trace_decompress_begin(void) {}
void ZSTD_trace_decompress_end(void) {}
void _Jv_RegisterClasses(void) {}
EOF
for so in libSM.so.6 libICE.so.6 libXrender.so.1; do
  gcc -shared -fPIC -Wl,-soname,"$so" -o "$so" stub_empty.c
done
gcc -shared -fPIC -Wl,-soname,libXfixes.so.3 -o libXfixes.so.3 stub_xext.c
gcc -shared -fPIC -Wl,-soname,libXi.so.6     -o libXi.so.6     stub_xext.c
gcc -shared -fPIC -Wl,-soname,libXt.so.6     -o libXt.so.6     stub_xext.c
for so in libGL.so.1 libOpenGL.so.0; do
  gcc -shared -fPIC -Wl,-soname,"$so" -o "$so" stub_gl.c
done
printf 'V_0.5.0 { global: xkb_*; };\n' > xkb.map
gcc -shared -fPIC -Wl,-soname,libxkbcommon.so.0 -o libxkbcommon.so.0 stub_xkb.c -Wl,--version-script=xkb.map
gcc -shared -fPIC -o libzstdtrace.so zstdtrace.c
echo "DONE — run:  $HERE/blenderpy $HERE/../film_main.py --benchmark"
