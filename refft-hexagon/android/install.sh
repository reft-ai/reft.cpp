#!/bin/sh
# =============================================================================
#  RefineFuture.AI - refft-hexagon installer (Android)
#
#  Fetches the latest calendar-versioned release (tag format: vYYYY.MM.DD.NN)
#  from https://github.com/refinefuture-ai/refft.cpp/releases, picks the Android
#  asset, downloads it, and installs it.
#
#  The script adapts to where it runs:
#    * on the device (Termux / adb shell)  -> installs locally
#    * on a host with adb                  -> unpacks locally and pushes to the
#                                             connected device via adb
#
#  Usage:
#      ./install.sh [options]
#      curl -fsSL <raw-url>/install.sh | sh
#
#  Options:
#      -t, --tag <tag>            Install a specific tag instead of the latest
#      -p, --prefix <dir>         Install prefix (default: auto, see below)
#      -v, --hexagon-version <vN> Hexagon arch variant, e.g. v73 / v81 (default: highest)
#          --mode <auto|device|adb>  Install target (default: auto)
#          --serial <serial>      adb device serial (adb mode)
#          --arch <arch>          Override CPU arch detection (aarch64)
#          --download-dir <dir>   Where to store the downloaded asset
#          --keep                 Keep the downloaded archive after install
#          --list                 List matching releases and assets, then exit
#          --token <token>        GitHub token (or set GITHUB_TOKEN) for rate limits
#      -h, --help                 Show this help
#
#  Default prefix:
#      device mode : $PREFIX/opt/refft-hexagon (Termux) or /data/local/tmp/refft-hexagon
#      adb mode    : /data/local/tmp/refft-hexagon on the device
# =============================================================================

set -eu

REPO="refinefuture-ai/refft.cpp"
API_BASE="https://api.github.com/repos/$REPO"
RELEASES_URL="https://github.com/$REPO/releases"
PKG_PREFIX="refft-hexagon"
DEVICE_DEFAULT_PREFIX="/data/local/tmp/refft-hexagon"

# Termux exports $PREFIX itself; remember it before we reuse the name.
TERMUX_PREFIX="${PREFIX:-}"

TAG=""
PREFIX=""
HEXAGON_VERSION="${REFFT_HEXAGON_VERSION:-}"
MODE="auto"
SERIAL="${ANDROID_SERIAL:-}"
ARCH_OVERRIDE=""
DOWNLOAD_DIR=""
KEEP_ARCHIVE=0
LIST_ONLY=0
TOKEN="${GITHUB_TOKEN:-}"

PLATFORM_NAME="android"
TMPDIR_SELF=""

# ---------------------------------------------------------------- utilities --

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

cleanup() {
    [ -n "$TMPDIR_SELF" ] && [ -d "$TMPDIR_SELF" ] && rm -rf "$TMPDIR_SELF"
    return 0
}
trap cleanup EXIT INT TERM

have() { command -v "$1" >/dev/null 2>&1; }

usage() {
    cat <<'EOF'
RefineFuture.AI - refft-hexagon installer (Android)

Installs the latest calendar-versioned release (tag format: vYYYY.MM.DD.NN)
from https://github.com/refinefuture-ai/refft.cpp/releases.

The script adapts to where it runs:
    on the device (Termux / adb shell) -> installs locally
    on a host with adb                 -> unpacks locally, pushes via adb

Usage:
    ./install.sh [options]
    curl -fsSL <url>/install.sh | sh
    curl -fsSL <url>/install.sh | sh -s -- --serial <device-serial>

Options:
    -t, --tag <tag>            Install a specific tag instead of the latest
    -p, --prefix <dir>         Install prefix (default: /data/local/tmp/refft-hexagon,
                               or $PREFIX/opt/refft-hexagon under Termux)
    -v, --hexagon-version <vN> Hexagon variant, e.g. v73 / v81 (default: highest)
        --mode <auto|device|adb>  Install target (default: auto)
        --serial <serial>      adb device serial (adb mode)
        --arch <arch>          Override CPU arch detection (aarch64)
        --download-dir <dir>   Where to store the downloaded asset
        --keep                 Keep the downloaded archive after install
        --list                 List matching releases and assets, then exit
        --token <token>        GitHub token (or set GITHUB_TOKEN) for rate limits
    -h, --help                 Show this help
EOF
    exit 0
}

# ------------------------------------------------------------ argument parse --

while [ $# -gt 0 ]; do
    case "$1" in
        -t|--tag)              TAG="${2:-}"; shift 2 ;;
        -p|--prefix)           PREFIX="${2:-}"; shift 2 ;;
        -v|--hexagon-version)  HEXAGON_VERSION="${2:-}"; shift 2 ;;
        --mode)                MODE="${2:-}"; shift 2 ;;
        --serial)              SERIAL="${2:-}"; shift 2 ;;
        --arch)                ARCH_OVERRIDE="${2:-}"; shift 2 ;;
        --download-dir)        DOWNLOAD_DIR="${2:-}"; shift 2 ;;
        --keep)                KEEP_ARCHIVE=1; shift ;;
        --list)                LIST_ONLY=1; shift ;;
        --token)               TOKEN="${2:-}"; shift 2 ;;
        -h|--help)             usage ;;
        *)                     die "unknown option: $1 (use --help)" ;;
    esac
done

# ------------------------------------------------------------------- http io --

http_get() {
    url="$1"
    if have curl; then
        if [ -n "$TOKEN" ]; then
            curl -fsSL -H "Accept: application/vnd.github+json" \
                 -H "Authorization: Bearer $TOKEN" "$url"
        else
            curl -fsSL -H "Accept: application/vnd.github+json" "$url"
        fi
    elif have wget; then
        if [ -n "$TOKEN" ]; then
            wget -qO- --header="Accept: application/vnd.github+json" \
                 --header="Authorization: Bearer $TOKEN" "$url"
        else
            wget -qO- --header="Accept: application/vnd.github+json" "$url"
        fi
    else
        die "neither curl nor wget is available"
    fi
}

http_download() {
    url="$1"; dest="$2"
    if have curl; then
        if [ -n "$TOKEN" ]; then
            curl -fL --progress-bar -H "Authorization: Bearer $TOKEN" -o "$dest" "$url"
        else
            curl -fL --progress-bar -o "$dest" "$url"
        fi
    else
        wget -q -O "$dest" "$url"
    fi
}

# --------------------------------------------------------------- release api --

TAG_REGEX='^v[0-9][0-9][0-9][0-9]\.[0-9][0-9]\.[0-9][0-9]\.[0-9][0-9]$'

list_release_tags() {
    page=1
    while [ "$page" -le 3 ]; do
        body=$(http_get "$API_BASE/releases?per_page=100&page=$page") || return 1
        names=$(printf '%s\n' "$body" \
                | grep -o '"tag_name"[[:space:]]*:[[:space:]]*"[^"]*"' \
                | sed 's/.*:[[:space:]]*"//; s/"$//')
        [ -z "$names" ] && break
        printf '%s\n' "$names"
        [ "$(printf '%s\n' "$names" | wc -l)" -lt 100 ] && break
        page=$((page + 1))
    done
}

latest_tag() {
    # vYYYY.MM.DD.NN -> numeric key YYYYMMDDNN, sorted descending.
    list_release_tags \
        | grep "$TAG_REGEX" \
        | while IFS= read -r t; do
              key=$(printf '%s\n' "$t" | sed 's/^v//; s/\.//g')
              printf '%s %s\n' "$key" "$t"
          done \
        | sort -k1,1nr \
        | head -n 1 \
        | cut -d' ' -f2
}

release_assets() {
    tag="$1"
    body=$(http_get "$API_BASE/releases/tags/$tag") \
        || die "release not found: $tag"

    if have jq; then
        printf '%s' "$body" | jq -r '
            .assets[] |
            [ .name, ((.digest // "") | sub("^sha256:"; "") | if . == "" then "-" else . end), .browser_download_url ]
            | @tsv'
    else
        # Flatten the JSON and split it into one object per line: break both on
        # object boundaries (},{) and on array openings ("assets":[{), so the
        # release header does not stick to the first asset. Asset objects are
        # the only ones carrying a browser_download_url.
        printf '%s' "$body" \
            | tr -d '\n' \
            | sed -e 's/\[[[:space:]]*{/[\
{/g' -e 's/},[[:space:]]*{/}\
{/g' \
            | grep 'browser_download_url' \
            | while IFS= read -r chunk; do
                  name=$(printf '%s' "$chunk" | grep -o '"name"[[:space:]]*:[[:space:]]*"[^"]*"' | tail -n1 | sed 's/.*:[[:space:]]*"//; s/"$//')
                  url=$(printf '%s' "$chunk" | grep -o '"browser_download_url"[[:space:]]*:[[:space:]]*"[^"]*"' | head -n1 | sed 's/.*:[[:space:]]*"//; s/"$//')
                  sha=$(printf '%s' "$chunk" | grep -o '"digest"[[:space:]]*:[[:space:]]*"sha256:[^"]*"' | head -n1 | sed 's/.*sha256://; s/"$//')
                  [ -z "$sha" ] && sha="-"
                  [ -n "$name" ] && [ -n "$url" ] && printf '%s\t%s\t%s\n' "$name" "$sha" "$url"
              done
    fi
}

# ------------------------------------------------------------ platform match --

running_on_android() {
    [ -n "${ANDROID_ROOT:-}" ] && return 0
    [ -d /system/bin ] && [ -f /system/build.prop ] && return 0
    case "$(uname -o 2>/dev/null || true)" in
        *Android*) return 0 ;;
    esac
    return 1
}

adb() {
    if [ -n "$SERIAL" ]; then
        command adb -s "$SERIAL" "$@"
    else
        command adb "$@"
    fi
}

resolve_mode() {
    case "$MODE" in
        device|adb) return ;;
        auto)
            if running_on_android; then
                MODE="device"
            elif have adb; then
                MODE="adb"
            else
                die "not running on Android and adb was not found; install platform-tools or use --mode device"
            fi
            ;;
        *) die "invalid --mode: $MODE (expected auto, device or adb)" ;;
    esac
}

detect_arch() {
    if [ -n "$ARCH_OVERRIDE" ]; then
        printf '%s\n' "$ARCH_OVERRIDE"
        return
    fi
    if [ "$MODE" = "adb" ]; then
        abi=$(adb shell getprop ro.product.cpu.abi 2>/dev/null | tr -d '\r\n' || true)
        case "$abi" in
            arm64*|aarch64*) printf 'aarch64\n'; return ;;
            armeabi*)        printf 'armv7\n';   return ;;
            x86_64)          printf 'x86_64\n';  return ;;
        esac
    fi
    case "$(uname -m)" in
        aarch64|arm64) printf 'aarch64\n' ;;
        x86_64|amd64)  printf 'x86_64\n' ;;
        *)             printf '%s\n' "$(uname -m)" ;;
    esac
}

arch_aliases() {
    case "$1" in
        aarch64|arm64) printf 'aarch64 arm64\n' ;;
        x86_64|amd64)  printf 'x86_64 amd64 x64\n' ;;
        *)             printf '%s\n' "$1" ;;
    esac
}

asset_rank() {
    name="$1"
    hv=$(printf '%s' "$name" | sed -n 's/.*-v\([0-9][0-9]*\).*/\1/p')
    [ -z "$hv" ] && hv=0
    case "$name" in
        *.tar.xz)       kind=4 ;;
        *.tar.gz|*.tgz) kind=3 ;;
        *.zip)          kind=2 ;;
        *)              kind=0 ;;
    esac
    printf '%s\n' $((hv * 10 + kind))
}

matches_platform() {
    name="$1"; arch="$2"
    case "$name" in
        "$PKG_PREFIX"*) : ;;
        *) return 1 ;;
    esac
    for a in $(arch_aliases "$arch"); do
        case "$name" in
            *"android-${a}"*) return 0 ;;
        esac
    done
    return 1
}

select_asset() {
    arch="$1"
    best_rank=-1
    best_line=""
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        name=$(printf '%s' "$line" | cut -f1)
        matches_platform "$name" "$arch" || continue
        if [ -n "$HEXAGON_VERSION" ]; then
            want=$(printf '%s' "$HEXAGON_VERSION" | sed 's/^[vV]//')
            case "$name" in
                *"-v${want}"*) : ;;
                *) continue ;;
            esac
        fi
        rank=$(asset_rank "$name")
        if [ "$rank" -gt "$best_rank" ]; then
            best_rank="$rank"
            best_line="$line"
        fi
    done
    [ -n "$best_line" ] && printf '%s\n' "$best_line"
}

# ------------------------------------------------------------------- install --

verify_sha256() {
    file="$1"; expected="$2"
    [ "$expected" = "-" ] && { warn "no checksum published for this asset, skipping verification"; return 0; }
    if have sha256sum; then actual=$(sha256sum "$file" | cut -d' ' -f1)
    elif have shasum;  then actual=$(shasum -a 256 "$file" | cut -d' ' -f1)
    elif have openssl; then actual=$(openssl dgst -sha256 "$file" | sed 's/.*= *//')
    else warn "no sha256 tool available, skipping verification"; return 0
    fi
    [ "$actual" = "$expected" ] || die "checksum mismatch (expected $expected, got $actual)"
    info "sha256 verified"
}

extract_archive() {
    file="$1"; dest="$2"
    mkdir -p "$dest"
    case "$file" in
        *.tar.xz)
            tar -xJf "$file" -C "$dest" 2>/dev/null \
                || { have xz || die "xz is required to extract $file"; xz -dc "$file" | tar -xf - -C "$dest"; } ;;
        *.tar.gz|*.tgz)  tar -xzf "$file" -C "$dest" ;;
        *.tar)           tar -xf  "$file" -C "$dest" ;;
        *.zip)
            if have unzip; then unzip -q "$file" -d "$dest"
            elif have bsdtar; then bsdtar -xf "$file" -C "$dest"
            else die "unzip is required to extract $file"; fi ;;
        *) die "unsupported archive format: $file" ;;
    esac
}

package_root() {
    dir="$1"
    count=$(ls -A "$dir" | wc -l)
    if [ "$count" -eq 1 ]; then
        only="$dir/$(ls -A "$dir")"
        [ -d "$only" ] && { printf '%s\n' "$only"; return; }
    fi
    printf '%s\n' "$dir"
}

find_bundled_installer() {
    root="$1"
    for candidate in install.sh setup.sh install run_install.sh scripts/install.sh; do
        if [ -f "$root/$candidate" ]; then
            printf '%s\n' "$root/$candidate"
            return 0
        fi
    done
    return 1
}

default_prefix() {
    if [ "$MODE" = "adb" ]; then
        printf '%s\n' "$DEVICE_DEFAULT_PREFIX"
    elif [ -n "$TERMUX_PREFIX" ] && [ -d "$TERMUX_PREFIX" ]; then
        printf '%s/opt/%s\n' "$TERMUX_PREFIX" "$PKG_PREFIX"
    elif [ -d /data/data/com.termux/files/usr ]; then
        printf '/data/data/com.termux/files/usr/opt/%s\n' "$PKG_PREFIX"
    else
        printf '%s\n' "$DEVICE_DEFAULT_PREFIX"
    fi
}

install_tree_local() {
    root="$1"; prefix="$2"
    mkdir -p "$prefix"
    ( cd "$root" && tar -cf - . ) | ( cd "$prefix" && tar -xf - )
    [ -d "$prefix/bin" ] && chmod +x "$prefix"/bin/* 2>/dev/null
    for f in "$prefix"/refft* "$prefix"/*.sh; do
        [ -f "$f" ] && chmod +x "$f" 2>/dev/null
    done
    return 0
}

install_via_adb() {
    # install_via_adb <package-root> <device-prefix> <installer-or-empty>
    root="$1"; prefix="$2"; installer="$3"

    log "Waiting for the device"
    adb wait-for-device || die "no device connected"
    model=$(adb shell getprop ro.product.model 2>/dev/null | tr -d '\r\n' || true)
    info "device   : ${model:-unknown} ${SERIAL:+($SERIAL)}"

    log "Pushing the package to $prefix"
    adb shell "rm -rf '$prefix' && mkdir -p '$prefix'" >/dev/null 2>&1 || \
        die "cannot create $prefix on the device"
    # Push the contents of the package root, not the root directory itself.
    for entry in "$root"/* "$root"/.[!.]*; do
        [ -e "$entry" ] || continue
        adb push "$entry" "$prefix/" >/dev/null || die "adb push failed for $entry"
    done
    adb shell "chmod -R 755 '$prefix' 2>/dev/null" >/dev/null 2>&1 || true

    if [ -n "$installer" ]; then
        rel=${installer#"$root"/}
        log "Running bundled installer on the device: $rel"
        adb shell "cd '$prefix' && REFFT_INSTALL_PREFIX='$prefix' PREFIX='$prefix' sh '$prefix/$rel'" \
            || die "the bundled installer failed on the device"
    fi
}

# ---------------------------------------------------------------------- main --

main() {
    have curl || have wget || die "curl or wget is required"
    have tar  || die "tar is required"

    resolve_mode
    ARCH=$(detect_arch)

    log "RefineFuture.AI refft-hexagon installer"
    info "platform : $PLATFORM_NAME/$ARCH"
    info "mode     : $MODE"
    info "source   : $RELEASES_URL"

    if [ -z "$TAG" ]; then
        log "Resolving the latest release (tag pattern vYYYY.MM.DD.NN)"
        TAG=$(latest_tag) || true
        [ -n "$TAG" ] || die "no release matching vYYYY.MM.DD.NN was found at $RELEASES_URL"
    fi
    info "release  : $TAG"

    ASSETS=$(release_assets "$TAG")
    [ -n "$ASSETS" ] || die "release $TAG has no downloadable assets"

    if [ "$LIST_ONLY" -eq 1 ]; then
        log "Releases matching vYYYY.MM.DD.NN (newest first)"
        list_release_tags | grep "$TAG_REGEX" \
            | while IFS= read -r t; do
                  printf '%s %s\n' "$(printf '%s' "$t" | sed 's/^v//; s/\.//g')" "$t"
              done | sort -k1,1nr | cut -d' ' -f2 | sed 's/^/    /'
        log "Assets of $TAG"
        printf '%s\n' "$ASSETS" | cut -f1 | sed 's/^/    /'
        exit 0
    fi

    CHOSEN=$(printf '%s\n' "$ASSETS" | select_asset "$ARCH")
    if [ -z "$CHOSEN" ]; then
        warn "no asset in $TAG matches $PLATFORM_NAME/$ARCH${HEXAGON_VERSION:+ (hexagon $HEXAGON_VERSION)}"
        info "available assets:"
        printf '%s\n' "$ASSETS" | cut -f1 | sed 's/^/      /'
        die "nothing to install"
    fi

    ASSET_NAME=$(printf '%s' "$CHOSEN" | cut -f1)
    ASSET_SHA=$(printf '%s' "$CHOSEN"  | cut -f2)
    ASSET_URL=$(printf '%s' "$CHOSEN"  | cut -f3)
    info "asset    : $ASSET_NAME"

    TMPDIR_SELF=$(mktemp -d "${TMPDIR:-/tmp}/refft-hexagon.XXXXXX")
    if [ -n "$DOWNLOAD_DIR" ]; then
        mkdir -p "$DOWNLOAD_DIR"
        ARCHIVE="$DOWNLOAD_DIR/$ASSET_NAME"
    else
        ARCHIVE="$TMPDIR_SELF/$ASSET_NAME"
    fi

    log "Downloading $ASSET_NAME"
    http_download "$ASSET_URL" "$ARCHIVE"
    verify_sha256 "$ARCHIVE" "$ASSET_SHA"

    log "Extracting package"
    EXTRACT_DIR="$TMPDIR_SELF/extract"
    extract_archive "$ARCHIVE" "$EXTRACT_DIR"
    ROOT=$(package_root "$EXTRACT_DIR")

    INSTALLER=""
    if INSTALLER=$(find_bundled_installer "$ROOT"); then :; else INSTALLER=""; fi

    [ -z "$PREFIX" ] && PREFIX=$(default_prefix)

    if [ "$MODE" = "adb" ]; then
        install_via_adb "$ROOT" "$PREFIX" "$INSTALLER"
        if [ -n "$INSTALLER" ]; then
            INSTALLED_AT="$PREFIX on the device (bundled installer)"
        else
            INSTALLED_AT="$PREFIX on the device"
        fi
    else
        if [ -n "$INSTALLER" ]; then
            log "Running bundled installer: $(basename "$INSTALLER")"
            chmod +x "$INSTALLER" 2>/dev/null || true
            (
                cd "$ROOT"
                REFFT_INSTALL_PREFIX="$PREFIX" PREFIX="$PREFIX" \
                REFFT_RELEASE_TAG="$TAG" REFFT_PLATFORM="$PLATFORM_NAME" \
                    sh "$INSTALLER"
            ) || die "the bundled installer failed"
            INSTALLED_AT="$PREFIX (bundled installer)"
        else
            log "No bundled installer found, extracting to $PREFIX"
            install_tree_local "$ROOT" "$PREFIX"
            INSTALLED_AT="$PREFIX"
        fi
    fi

    if [ "$KEEP_ARCHIVE" -eq 1 ] && [ -z "$DOWNLOAD_DIR" ]; then
        cp "$ARCHIVE" "./$ASSET_NAME"
        ARCHIVE="./$ASSET_NAME"
    fi
    { [ -n "$DOWNLOAD_DIR" ] || [ "$KEEP_ARCHIVE" -eq 1 ]; } && info "archive kept at: $ARCHIVE"

    # Name the entry point the package actually ships, if there is one.
    BIN_NAME=""
    if [ -d "$ROOT/bin" ]; then
        BIN_NAME=$(ls -1 "$ROOT/bin" 2>/dev/null | head -n 1)
    fi

    printf '\n'
    log "Installation complete"
    info "release   : $TAG"
    info "package   : $ASSET_NAME"
    info "installed : $INSTALLED_AT"
    if [ "$MODE" = "adb" ]; then
        if [ -n "$BIN_NAME" ]; then
            info "run it with: adb shell \"cd $PREFIX && LD_LIBRARY_PATH=$PREFIX/lib ./bin/$BIN_NAME --help\""
        fi
    else
        info "add to PATH: export PATH=\"$PREFIX/bin:\$PATH\""
        [ -d "$PREFIX/lib" ] && info "shared libs: export LD_LIBRARY_PATH=\"$PREFIX/lib:\$LD_LIBRARY_PATH\""
    fi
}

main
