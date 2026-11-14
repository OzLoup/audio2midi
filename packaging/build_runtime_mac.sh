#!/bin/bash
# Builds audio2midi.app (+ .dmg) for macOS arm64: portable Python 3.12 (python-build-standalone) inside the bundle.
#
#   dist/audio2midi.app/Contents/Resources/{runtime/main, src, bin, launch.pyw}
#   dist/audio2midi.app/Contents/MacOS/audio2midi   (launcher: runtime/main/bin/python -s launch.pyw)
#
# Run from anywhere:  packaging/build_runtime_mac.sh [--app-only] [--no-dmg]
#   SIGN_IDENTITY="AetherIO Local"            sign with that identity (default: ad-hoc "-")
#   SIGN_IDENTITY="Developer ID Application: Name (TEAMID)" NOTARY_PROFILE=a2m   also notarize + staple
#       (create the profile once: xcrun notarytool store-credentials a2m --apple-id ... --team-id ...)
# Without a Developer ID the app is NOT notarized: it opens without prompts on the Mac that built it; on
# another Mac use right-click > Open, or `xattr -cr audio2midi.app`.
#
# Needs internet the first time (Python, wheels). ffmpeg: bin/ffmpeg if present, else the static one from imageio-ffmpeg.
set -euo pipefail
export PYTHONNOUSERSITE=1

PBS_TAG="20261110"
PY_VER="3.12.16"
ADTOF_COMMIT="85c192e"
ADTOF_REPO="https://github.com/xavriley/ADTOF-pytorch.git"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
APP="$REPO/dist/audio2midi.app"
RES="$APP/Contents/Resources"
CACHE="$REPO/packaging/cache"
IDENTITY="${SIGN_IDENTITY:--}"
APP_ONLY=0; DMG=1
for a in "$@"; do case "$a" in --app-only) APP_ONLY=1;; --no-dmg) DMG=0;; *) echo "unknown option $a"; exit 2;; esac; done
step() { printf '\n== %s\n' "$1"; }
mkdir -p "$CACHE"

copy_app() {
  step "app"
  rm -rf "$RES/src" "$RES/bin"
  mkdir -p "$RES" "$APP/Contents/MacOS"
  cp -R src "$RES/src"; cp -R bin "$RES/bin"; cp launch.pyw "$RES/launch.pyw"
  find "$RES/src" -name __pycache__ -type d -prune -exec rm -rf {} +
  cat > "$APP/Contents/MacOS/audio2midi" <<'LAUNCH'
#!/bin/bash
RES="$(cd "$(dirname "$0")/../Resources" && pwd)"
export PYTHONNOUSERSITE=1
cd "$RES"
exec "$RES/runtime/main/bin/python3" -s launch.pyw
LAUNCH
  chmod +x "$APP/Contents/MacOS/audio2midi"
  cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>audio2midi</string>
  <key>CFBundleDisplayName</key><string>audio2midi</string>
  <key>CFBundleIdentifier</key><string>com.ozloup.audio2midi</string>
  <key>CFBundleExecutable</key><string>audio2midi</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
}

sign_all() {
  step "sign ($IDENTITY)"
  local ent="$CACHE/entitlements.plist" opts=(--force --sign "$IDENTITY")
  cat > "$ent" <<'ENT'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>com.apple.security.cs.allow-unsigned-executable-memory</key><true/>
  <key>com.apple.security.cs.disable-library-validation</key><true/>
  <key>com.apple.security.cs.allow-jit</key><true/>
</dict></plist>
ENT
  [ "$IDENTITY" != "-" ] && opts+=(--options runtime --timestamp --entitlements "$ent")
  # every Mach-O file (.so, .dylib, executables), deepest first, then the bundle itself
  find "$APP" -type f -print0 | while IFS= read -r -d '' f; do
    if file -b "$f" | grep -q "Mach-O"; then echo "$f"; fi
  done | awk '{ print length($0) "\t" $0 }' | sort -rn | cut -f2- | while IFS= read -r f; do
    codesign "${opts[@]}" "$f" 2>&1 | grep -v "replacing existing signature" || true
  done
  codesign "${opts[@]}" "$APP"
  codesign --verify --deep --strict "$APP" && echo "signature ok"
}

notarize() {
  if [ -z "${NOTARY_PROFILE:-}" ] || [[ "$IDENTITY" != Developer\ ID* ]]; then
    echo "(no Developer ID + NOTARY_PROFILE: skipping notarization)"; return
  fi
  step "notarize"
  local zip="$CACHE/audio2midi.zip"
  ditto -c -k --keepParent "$APP" "$zip"
  xcrun notarytool submit "$zip" --keychain-profile "$NOTARY_PROFILE" --wait
  xcrun stapler staple "$APP"
}

if [ "$APP_ONLY" = 1 ]; then
  [ -x "$RES/runtime/main/bin/python3" ] || { echo "No build yet: run without --app-only first"; exit 1; }
  copy_app; sign_all; echo "App refreshed in $APP"; exit 0
fi

# ---- 1. portable Python, verified against the published checksum
step "Python $PY_VER"
[ "$(uname -m)" = "arm64" ] || { echo "This build is for Apple Silicon (arm64)"; exit 1; }
NAME="cpython-$PY_VER+$PBS_TAG-aarch64-apple-darwin-install_only.tar.gz"
BASE="https://github.com/astral-sh/python-build-standalone/releases/download/$PBS_TAG"
[ -f "$CACHE/$NAME" ] || curl -fL "$BASE/${NAME//+/%2B}" -o "$CACHE/$NAME"
WANT="$(curl -fsL "$BASE/SHA256SUMS" | grep -F "$NAME" | awk '{print $1}' | head -1)"
[ -n "$WANT" ] || { echo "No checksum published for $NAME"; exit 1; }
GOT="$(shasum -a 256 "$CACHE/$NAME" | awk '{print $1}')"
[ "$GOT" = "$WANT" ] || { rm -f "$CACHE/$NAME"; echo "Checksum mismatch ($GOT vs $WANT)"; exit 1; }
echo "checksum ok (${GOT:0:12}...)"

# ---- 2. main runtime
step "runtime/main"
rm -rf "$APP"; mkdir -p "$RES/runtime" "$CACHE/extract"
rm -rf "$CACHE/extract"/*; tar -xzf "$CACHE/$NAME" -C "$CACHE/extract"
mv "$CACHE/extract/python" "$RES/runtime/main"
PY="$RES/runtime/main/bin/python3"
# one package at a time would be slower; PyTorch is the memory hog (a build died of that), so no parallel jobs
"$PY" -s -m pip install --no-warn-script-location --disable-pip-version-check --no-cache-dir -r requirements-mac.txt
if [ -d third_party/ADTOF-pytorch ]; then
  "$PY" -s -m pip install --no-warn-script-location --disable-pip-version-check --no-deps third_party/ADTOF-pytorch
else
  "$PY" -s -m pip install --no-warn-script-location --disable-pip-version-check --no-deps "git+$ADTOF_REPO@$ADTOF_COMMIT"
fi

# ---- ffmpeg: the static arm64 binary that imageio-ffmpeg ships (Homebrew's depends on dylibs outside the bundle)
if [ ! -x bin/ffmpeg ]; then
  mkdir -p bin
  cp "$("$PY" -s -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" bin/ffmpeg
  chmod +x bin/ffmpeg
fi

# ---- 3. tidy: pip/audio-separator launchers keep absolute paths and break once moved (the app calls python -c/-m)
step "tidy"
find "$RES/runtime/main/bin" \( -type f -o -type l \) ! -name 'python*' -delete 2>/dev/null || true

# ---- 4. app, signature, verification
copy_app
step "verify"
"$PY" -s -m pip check
"$PY" -s -c "import torch, onnxruntime, librosa, pretty_midi, adtof_pytorch, audio_separator, PySide6; print('ok, torch', torch.__version__, 'mps', torch.backends.mps.is_available())"
sign_all
notarize

# ---- 5. dmg
if [ "$DMG" = 1 ]; then
  step "dmg"
  STAGE="$CACHE/dmg"; rm -rf "$STAGE"; mkdir -p "$STAGE"
  cp -R "$APP" "$STAGE/"; ln -s /Applications "$STAGE/Applications"
  hdiutil create -volname audio2midi -srcfolder "$STAGE" -ov -format UDZO "$REPO/dist/audio2midi.dmg" >/dev/null
  [[ "$IDENTITY" == Developer\ ID* ]] && codesign --force --sign "$IDENTITY" "$REPO/dist/audio2midi.dmg"
  echo "dmg: $REPO/dist/audio2midi.dmg"
fi
du -sh "$APP"
