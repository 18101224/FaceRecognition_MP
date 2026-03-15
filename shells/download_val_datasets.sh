#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-val_datasets}"
mkdir -p "$ROOT"

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "Missing command: $1"; exit 1; }; }
need_cmd tar
need_cmd unzip || true
need_cmd curl || need_cmd wget

dl() {
  local url="$1"
  local out="$2"
  echo "[DL] $url"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 5 --retry-delay 2 -o "$out" "$url"
  else
    wget -O "$out" "$url"
  fi
}

LFW_DIR="$ROOT/lfw"
mkdir -p "$LFW_DIR"
cd "$LFW_DIR"

LFW_ARCHIVE_URL="https://ndownloader.figshare.com/files/5976015"
LFW_ARCHIVE_NAME="lfw-funneled.tgz"

dl "$LFW_ARCHIVE_URL" "$LFW_ARCHIVE_NAME"
dl "https://ndownloader.figshare.com/files/5976006" "pairs.txt"

echo "[EXTRACT] $LFW_ARCHIVE_NAME"
tar -xzf "$LFW_ARCHIVE_NAME"
rm -f "$LFW_ARCHIVE_NAME"

cd - >/dev/null

XQLFW_DIR="$ROOT/xqlfw"
mkdir -p "$XQLFW_DIR"
cd "$XQLFW_DIR"

dl "https://github.com/Martlgap/xqlfw/releases/download/1.0/xqlfw_aligned_112.zip" "xqlfw_aligned_112.zip"
dl "https://github.com/Martlgap/xqlfw/releases/download/1.0/xqlfw_pairs.txt" "xqlfw_pairs.txt"

echo "[EXTRACT] xqlfw_aligned_112.zip"
unzip -o "xqlfw_aligned_112.zip"
rm -f "xqlfw_aligned_112.zip"

cd - >/dev/null

echo "DONE. Root: $ROOT"
echo " - LFW:   $ROOT/lfw"
echo " - XQLFW: $ROOT/xqlfw"
