#!/bin/sh
# =============================================================================
#  RefineFuture.AI - refft-hexagon installer (Linux)
#
#  Fetches the latest calendar-versioned release (tag format: vYYYY.MM.DD.NN)
#  from https://github.com/refinefuture-ai/refft.cpp/releases, picks the asset
#  that matches this machine, downloads it, and installs it.
#
#  Usage:
#      ./install.sh [options]
#      curl -fsSL <raw-url>/install.sh | sh
#
#  Options:
#      -t, --tag <tag>            Install a specific tag instead of the latest
#      -p, --prefix <dir>         Install prefix (default: auto)
#      -v, --hexagon-version <vN> Hexagon arch variant, e.g. v73 / v81 (default: highest)
#          --arch <arch>          Override CPU arch detection (arm64 / x86_64)
#          --download-dir <dir>   Where to store the downloaded asset
#          --prefer-deb           Prefer a .deb asset over a tarball
#          --keep                 Keep the downloaded archive after install
#          --list                 List matching releases and assets, then exit
#          --token <token>        GitHub token (or set GITHUB_TOKEN) for rate limits
#      -h, --help                 Show this help
# =============================================================================

set -eu

REPO="refinefuture-ai/refft.cpp"
API_BASE="https://api.github.com/repos/$REPO"
RELEASES_URL="https://github.com/$REPO/releases"
PKG_PREFIX="refft-hexagon"

TAG=""
PREFIX=""
HEXAGON_VERSION="${REFFT_HEXAGON_VERSION:-}"
ARCH_OVERRIDE=""
DOWNLOAD_DIR=""
PREFER_DEB=0
KEEP_ARCHIVE=0
LIST_ONLY=0
TOKEN="${GITHUB_TOKEN:-}"

PLATFORM_NAME="linux"
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

need() {
    have "$1" || die "required command not found: $1"
}

usage() {
    cat <<'EOF'
RefineFuture.AI - refft-hexagon installer (Linux)

Installs the latest calendar-versioned release (tag format: vYYYY.MM.DD.NN)
from https://github.com/refinefuture-ai/refft.cpp/releases.

Usage:
    ./install.sh [options]
    curl -fsSL <url>/install.sh | sh
    curl -fsSL <url>/install.sh | sh -s -- --prefix /opt/refft-hexagon

Options:
    -t, --tag <tag>            Install a specific tag instead of the latest
    -p, --prefix <dir>         Install prefix (default: /opt/refft-hexagon as
                               root, ~/.local/share/refft-hexagon otherwise)
    -v, --hexagon-version <vN> Hexagon variant, e.g. v73 / v81 (default: highest)
        --arch <arch>          Override CPU arch detection (arm64 / x86_64)
        --download-dir <dir>   Where to store the downloaded asset
        --prefer-deb           Prefer a .deb asset over a tarball
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
        --arch)                ARCH_OVERRIDE="${2:-}"; shift 2 ;;
        --download-dir)        DOWNLOAD_DIR="${2:-}"; shift 2 ;;
        --prefer-deb)          PREFER_DEB=1; shift ;;
        --keep)                KEEP_ARCHIVE=1; shift ;;
        --list)                LIST_ONLY=1; shift ;;
        --token)               TOKEN="${2:-}"; shift 2 ;;
        -h|--help)             usage ;;
        *)                     die "unknown option: $1 (use --help)" ;;
    esac
done

# ------------------------------------------------------------------- http io --

http_get() {
    # http_get <url>  -> body on stdout
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
    # http_download <url> <dest>
    url="$1"; dest="$2"
    if have curl; then
        if [ -n "$TOKEN" ]; then
            curl -fL --progress-bar -H "Authorization: Bearer $TOKEN" -o "$dest" "$url"
        else
            curl -fL --progress-bar -o "$dest" "$url"
        fi
    else
        wget -q --show-progress -O "$dest" "$url"
    fi
}

# --------------------------------------------------------------- release api --

# Tag format: vYYYY.MM.DD.NN  (NN = the NN-th release of that day)
TAG_REGEX='^v[0-9][0-9][0-9][0-9]\.[0-9][0-9]\.[0-9][0-9]\.[0-9][0-9]$'

list_release_tags() {
    # Walk up to 3 pages of releases and emit every tag name.
    page=1
    while [ "$page" -le 3 ]; do
        body=$(http_get "$API_BASE/releases?per_page=100&page=$page") || return 1
        names=$(printf '%s\n' "$body" \
                | grep -o '"tag_name"[[:space:]]*:[[:space:]]*"[^"]*"' \
                | sed 's/.*:[[:space:]]*"//; s/"$//')
        [ -z "$names" ] && break
        printf '%s\n' "$names"
        # A short page means we reached the end.
        [ "$(printf '%s\n' "$names" | wc -l)" -lt 100 ] && break
        page=$((page + 1))
    done
}

latest_tag() {
    # Keep only vYYYY.MM.DD.NN tags, build a numeric sort key YYYYMMDDNN,
    # sort descending (year, month, day, then release counter) and take the top.
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
    # release_assets <tag> -> "<name>\t<sha256|->\t<url>" per line
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

detect_arch() {
    if [ -n "$ARCH_OVERRIDE" ]; then
        printf '%s\n' "$ARCH_OVERRIDE"
        return
    fi
    case "$(uname -m)" in
        aarch64|arm64)      printf 'arm64\n' ;;
        x86_64|amd64)       printf 'x86_64\n' ;;
        *)                  printf '%s\n' "$(uname -m)" ;;
    esac
}

arch_aliases() {
    # Asset names are not fully consistent across releases, so try every spelling.
    case "$1" in
        arm64|aarch64)  printf 'arm64 aarch64\n' ;;
        x86_64|amd64)   printf 'x86_64 amd64 x64\n' ;;
        *)              printf '%s\n' "$1" ;;
    esac
}

os_aliases() {
    # On Linux the packages are published with an "ubuntu-" prefix.
    printf 'ubuntu linux\n'
}

asset_rank() {
    # Higher is better: hexagon variant (v81 > v73) dominates, then archive kind.
    name="$1"
    hv=$(printf '%s' "$name" | sed -n 's/.*-v\([0-9][0-9]*\).*/\1/p')
    [ -z "$hv" ] && hv=0
    case "$name" in
        *.tar.xz)            kind=4 ;;
        *.tar.gz|*.tgz)      kind=3 ;;
        *.zip)               kind=2 ;;
        *.deb)               kind=$([ "$PREFER_DEB" -eq 1 ] && echo 9 || echo 1) ;;
        *)                   kind=0 ;;
    esac
    printf '%s\n' $((hv * 10 + kind))
}

matches_platform() {
    name="$1"; arch="$2"
    case "$name" in
        "$PKG_PREFIX"*) : ;;
        *) return 1 ;;
    esac
    for os in $(os_aliases); do
        for a in $(arch_aliases "$arch"); do
            case "$name" in
                *"${os}-${a}"*) return 0 ;;
            esac
        done
    done
    return 1
}

select_asset() {
    # select_asset <arch> <assets...>  -> chosen "<name>\t<sha>\t<url>"
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
    if have sha256sum;   then actual=$(sha256sum "$file" | cut -d' ' -f1)
    elif have shasum;    then actual=$(shasum -a 256 "$file" | cut -d' ' -f1)
    elif have openssl;   then actual=$(openssl dgst -sha256 "$file" | sed 's/.*= *//')
    else warn "no sha256 tool available, skipping verification"; return 0
    fi
    [ "$actual" = "$expected" ] || die "checksum mismatch (expected $expected, got $actual)"
    info "sha256 verified"
}

extract_archive() {
    # extract_archive <file> <dest-dir>
    file="$1"; dest="$2"
    mkdir -p "$dest"
    case "$file" in
        *.tar.xz)
            have xz || have tar || die "tar/xz not available"
            tar -xJf "$file" -C "$dest" 2>/dev/null \
                || { xz -dc "$file" | tar -xf - -C "$dest"; } ;;
        *.tar.gz|*.tgz)  tar -xzf "$file" -C "$dest" ;;
        *.tar.bz2)       tar -xjf "$file" -C "$dest" ;;
        *.tar)           tar -xf  "$file" -C "$dest" ;;
        *.zip)
            if have unzip; then unzip -q "$file" -d "$dest"
            elif have bsdtar; then bsdtar -xf "$file" -C "$dest"
            else die "unzip is required to extract $file"; fi ;;
        *) die "unsupported archive format: $file" ;;
    esac
}

package_root() {
    # A tarball usually holds a single top-level directory; use it as the root.
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
    if [ "$(id -u)" = "0" ]; then
        printf '/opt/%s\n' "$PKG_PREFIX"
    else
        printf '%s/.local/share/%s\n' "$HOME" "$PKG_PREFIX"
    fi
}

install_deb() {
    file="$1"
    log "Installing Debian package"
    if [ "$(id -u)" = "0" ]; then
        dpkg -i "$file" || { apt-get install -f -y && dpkg -i "$file"; }
    elif have sudo; then
        sudo dpkg -i "$file" || { sudo apt-get install -f -y && sudo dpkg -i "$file"; }
    else
        die "root privileges are required to install $file"
    fi
}

install_tree() {
    # install_tree <package-root> <prefix>
    root="$1"; prefix="$2"
    mkdir -p "$prefix"
    ( cd "$root" && tar -cf - . ) | ( cd "$prefix" && tar -xf - )
    # Make anything under bin/ and any bare ELF launcher executable.
    [ -d "$prefix/bin" ] && chmod +x "$prefix"/bin/* 2>/dev/null
    for f in "$prefix"/refft* "$prefix"/*.sh; do
        [ -f "$f" ] && chmod +x "$f" 2>/dev/null
    done
    return 0
}

# ---------------------------------------------------------------------- main --

main() {
    have curl || have wget || die "curl or wget is required"
    need tar

    ARCH=$(detect_arch)
    log "RefineFuture.AI refft-hexagon installer"
    info "platform : $PLATFORM_NAME/$ARCH"
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

    [ -z "$PREFIX" ] && PREFIX=$(default_prefix)

    case "$ASSET_NAME" in
        *.deb)
            install_deb "$ARCHIVE"
            INSTALLED_AT="system (dpkg)"
            ;;
        *)
            log "Extracting package"
            EXTRACT_DIR="$TMPDIR_SELF/extract"
            extract_archive "$ARCHIVE" "$EXTRACT_DIR"
            ROOT=$(package_root "$EXTRACT_DIR")

            if INSTALLER=$(find_bundled_installer "$ROOT"); then
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
                install_tree "$ROOT" "$PREFIX"
                INSTALLED_AT="$PREFIX"
            fi
            ;;
    esac

    if [ -n "$DOWNLOAD_DIR" ] || [ "$KEEP_ARCHIVE" -eq 1 ]; then
        [ "$KEEP_ARCHIVE" -eq 1 ] && [ -z "$DOWNLOAD_DIR" ] && {
            cp "$ARCHIVE" "./$ASSET_NAME"; ARCHIVE="./$ASSET_NAME"
        }
        info "archive kept at: $ARCHIVE"
    fi

    printf '\n'
    log "Installation complete"
    info "release   : $TAG"
    info "package   : $ASSET_NAME"
    info "installed : $INSTALLED_AT"
    if [ -d "$PREFIX/bin" ]; then
        info "add to PATH: export PATH=\"$PREFIX/bin:\$PATH\""
    elif [ -d "$PREFIX" ]; then
        info "add to PATH: export PATH=\"$PREFIX:\$PATH\""
    fi
    [ -d "$PREFIX/lib" ] && info "shared libs: export LD_LIBRARY_PATH=\"$PREFIX/lib:\$LD_LIBRARY_PATH\""
    return 0
}

main
