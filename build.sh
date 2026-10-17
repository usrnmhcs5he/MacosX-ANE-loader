#!/bin/bash
# Builds "build/ANE Load.app". Needs an Apple Silicon Mac with the Xcode
# Command Line Tools (xcode-select --install). Full Xcode is not required.
set -euo pipefail
cd "$(dirname "$0")"

[[ "$(uname -m)" == "arm64" ]] || { echo "This needs an Apple Silicon Mac."; exit 1; }
xcrun --find swiftc >/dev/null 2>&1 || { echo "Install the Command Line Tools first: xcode-select --install"; exit 1; }

APP="build/ANE Load.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

xcrun swiftc -O -parse-as-library \
    -sdk "$(xcrun --sdk macosx --show-sdk-path)" \
    -target arm64-apple-macos14.0 \
    Sources/ANELoadApp.swift \
    -o "$APP/Contents/MacOS/ANELoad"

cp Info.plist "$APP/Contents/Info.plist"
cp -R Model/AneStress.mlpackage "$APP/Contents/Resources/"
codesign --force --sign - "$APP"

echo "Built: $APP"
echo "Run:   open \"$APP\""
