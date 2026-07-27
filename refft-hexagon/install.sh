#!/bin/sh
# =============================================================================
#  RefineFuture.AI - refft-hexagon universal installer
#
#  Detects the platform and hands over to the matching installer:
#      Android (Termux / adb shell / host with adb) -> android/install.sh
#      Linux                                        -> linux/install.sh
#      Windows                                      -> windows/install.ps1
#
#  One-liner:
#      curl -fsSL https://raw.githubusercontent.com/refinefuture-ai/refft.cpp/main/refft-hexagon/install.sh | sh
#
#  With options (everything after `--` is forwarded to the platform installer):
#      curl -fsSL <url>/install.sh | sh -s -- --prefix /opt/refft-hexagon
#
#  Environment:
#      REFFT_INSTALL_REF   Branch/tag to fetch the platform installer from (default: main)
#      REFFT_PLATFORM      Force a platform: android | linux
# =============================================================================

set -eu

REPO="refinefuture-ai/refft.cpp"
REF="${REFFT_INSTALL_REF:-main}"
RAW_BASE="https://raw.githubusercontent.com/$REPO/$REF/refft-hexagon"

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

adb_device_present() {
    have adb || return 1
    adb devices 2>/dev/null | sed '1d' | grep -qw device
}

detect_platform() {
    [ -n "${REFFT_PLATFORM:-}" ] && { printf '%s\n' "$REFFT_PLATFORM"; return; }

    # Running on an Android device (Termux or adb shell).
    if [ -n "${ANDROID_ROOT:-}" ] || { [ -d /system/bin ] && [ -f /system/build.prop ]; }; then
        printf 'android\n'; return
    fi
    case "$(uname -o 2>/dev/null || true)" in
        *Android*) printf 'android\n'; return ;;
    esac

    case "$(uname -s 2>/dev/null || true)" in
        Linux)
            # arm64 Linux boxes (e.g. Snapdragon dev kits) run the packages
            # natively. On x86_64 there is no native Hexagon package, so a
            # connected Android device is the intended target.
            case "$(uname -m)" in
                aarch64|arm64)
                    if adb_device_present; then
                        printf 'a connected Android device was found; to install onto it run:\n    REFFT_PLATFORM=android sh install.sh\n' >&2
                    fi
                    printf 'linux\n'
                    ;;
                *)
                    if adb_device_present; then
                        printf 'no native Hexagon package exists for %s Linux; targeting the connected Android device\n' "$(uname -m)" >&2
                        printf 'android\n'
                    else
                        printf 'linux\n'
                    fi
                    ;;
            esac
            ;;
        Darwin)
            die "macOS is not supported by refft-hexagon (Hexagon NPU packages are Android/Linux/Windows only)"
            ;;
        MINGW*|MSYS*|CYGWIN*)
            die "on Windows, run the PowerShell installer instead:
    irm $RAW_BASE/windows/install.ps1 | iex"
            ;;
        *)
            die "unsupported operating system: $(uname -s 2>/dev/null || echo unknown)"
            ;;
    esac
}

script_dir() {
    # Only meaningful when the script is executed from a checkout; when piped
    # through `sh` there is no file to resolve.
    case "${0:-}" in
        ''|sh|-sh|bash|-bash|dash|-) return 1 ;;
    esac
    [ -f "$0" ] || return 1
    dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd) || return 1
    printf '%s\n' "$dir"
}

main() {
    PLATFORM=$(detect_platform)
    log "Detected platform: $PLATFORM"

    if dir=$(script_dir) && [ -f "$dir/$PLATFORM/install.sh" ]; then
        log "Using local installer: $dir/$PLATFORM/install.sh"
        exec sh "$dir/$PLATFORM/install.sh" "$@"
    fi

    URL="$RAW_BASE/$PLATFORM/install.sh"
    log "Fetching $URL"
    TMPFILE=$(mktemp "${TMPDIR:-/tmp}/refft-hexagon-install.XXXXXX")
    if have curl; then
        curl -fsSL "$URL" -o "$TMPFILE" || die "failed to download $URL"
    elif have wget; then
        wget -qO "$TMPFILE" "$URL" || die "failed to download $URL"
    else
        die "curl or wget is required"
    fi
    [ -s "$TMPFILE" ] || die "downloaded installer is empty: $URL"

    set +e
    sh "$TMPFILE" "$@"
    status=$?
    set -e
    rm -f "$TMPFILE"
    exit $status
}

main "$@"
