# refft-hexagon installers

One-command installers for the **refft-hexagon** packages published at
<https://github.com/refinefuture-ai/refft.cpp/releases>.

Each script resolves the newest calendar-versioned release on its own — no
version pinning required.

## Quick start

**Linux**

```sh
curl -fsSL https://raw.githubusercontent.com/refinefuture-ai/refft.cpp/main/refft-hexagon/install.sh | sh
```

**Android** (from a host with `adb`, or directly in Termux on the device)

```sh
curl -fsSL https://raw.githubusercontent.com/refinefuture-ai/refft.cpp/main/refft-hexagon/android/install.sh | sh
```

**Windows** (PowerShell)

```powershell
irm https://raw.githubusercontent.com/refinefuture-ai/refft.cpp/main/refft-hexagon/install.ps1 | iex
```

Passing options through a pipe:

```sh
curl -fsSL <url>/install.sh | sh -s -- --prefix /opt/refft-hexagon --hexagon-version v81
```

```powershell
& ([scriptblock]::Create((irm <url>/install.ps1))) -Prefix 'C:\refft-hexagon' -AddToPath
```

## Layout

| Path | Purpose |
| --- | --- |
| `install.sh` | Universal entry point: detects the platform, delegates to the right script |
| `install.ps1` | Universal entry point for PowerShell (delegates to `windows/install.ps1`) |
| `linux/install.sh` | Linux installer |
| `android/install.sh` | Android installer (on-device or host + `adb`) |
| `windows/install.ps1` | Windows installer |

The platform scripts are self-contained: each can be downloaded and run on its
own, without the rest of the repository.

## What the scripts do

1. **List the releases** through the GitHub API.
2. **Keep only calendar tags** matching `vYYYY.MM.DD.NN` (e.g. `v2026.07.26.00`);
   any other tag in the repository is ignored.
3. **Sort them** by year, month, day and finally by `NN` — the counter of the
   n-th release of that day — and take the newest one.
4. **Pick the asset** for the current platform and CPU architecture. Assets are
   named `refft-hexagon_<os>-<arch>-<hexagon-variant>.<ext>`, for example:
   - `refft-hexagon_android-aarch64-v81.tar.xz`
   - `refft-hexagon_ubuntu-arm64-v73.tar.xz`
   - `refft-hexagon_windows-arm64-v81.zip`

   When a release ships several Hexagon variants (`v73`, `v75`, `v81`, …) the
   highest one is used unless `--hexagon-version` / `-HexagonVersion` says
   otherwise.
5. **Download and verify** the asset against the SHA-256 digest published by
   GitHub.
6. **Install it**: if the package contains its own installer
   (`install.sh` / `setup.sh` on Unix, `install.ps1` / `install.cmd` on Windows)
   that script is executed with `REFFT_INSTALL_PREFIX` set; otherwise the
   package is simply unpacked into the install prefix.
7. **Print `Installation complete`** together with the release, the package and
   where it landed.

## Platform detection

`install.sh` chooses the target as follows:

| Host | Target |
| --- | --- |
| Termux / `adb shell` on a device | `android`, installed locally on the device |
| arm64 Linux | `linux`, installed natively |
| x86_64 Linux with an `adb` device connected | `android`, pushed to the device |
| x86_64 Linux without a device | `linux` (reports the available assets if none match) |
| Windows / Git-Bash | points at the PowerShell installer |

Override it with `REFFT_PLATFORM=android|linux`.

## Default install prefixes

| Platform | Prefix |
| --- | --- |
| Linux (root) | `/opt/refft-hexagon` |
| Linux (user) | `~/.local/share/refft-hexagon` |
| Android via `adb` | `/data/local/tmp/refft-hexagon` on the device |
| Android in Termux | `$PREFIX/opt/refft-hexagon` |
| Windows | `%LOCALAPPDATA%\Programs\refft-hexagon` |

## Common options

Unix (`--help` for the full list):

```
-t, --tag <tag>              install a specific release instead of the latest
-p, --prefix <dir>           install prefix
-v, --hexagon-version <vN>   Hexagon variant, e.g. v73 / v81
    --arch <arch>            override CPU architecture detection
    --mode <auto|device|adb>  (Android only) install locally or push via adb
    --serial <serial>         (Android only) target a specific adb device
    --list                   show matching releases and assets, then exit
    --keep                   keep the downloaded archive
    --token <token>          GitHub token (or $GITHUB_TOKEN) for API rate limits
```

Windows: `-Tag`, `-Prefix`, `-HexagonVersion`, `-Arch`, `-List`, `-Keep`,
`-Token`, `-AddToPath`.

## Requirements

- **Linux / Android**: `curl` or `wget`, `tar` (plus `xz` for `.tar.xz` if your
  `tar` lacks it), and `adb` when installing onto a device from a host. `jq` is
  used when available; a pure-shell JSON fallback runs otherwise.
- **Windows**: PowerShell 5.1 or newer, and `tar.exe` (bundled since
  Windows 10 1803) for `.tar.*` packages.
