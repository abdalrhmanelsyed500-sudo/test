#!/bin/bash
# PALACE — run film_main.py with the in-repo bpy installation.
# Usage:  palace_film/_env/blenderpy film_main.py [--detect|--build-only|--benchmark|--render|--audio|--assemble|--all] [--preset A|B|C]
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env"
export PYTHONPATH="$HERE/pylibs:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$HERE/stublibs:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$HERE/stublibs/libzstdtrace.so${LD_PRELOAD:+:$LD_PRELOAD}"
export XDG_CACHE_HOME="$HERE/cache" XDG_DATA_HOME="$HERE/cache" XDG_CONFIG_HOME="$HERE/cache"
exec python3 "$@"
