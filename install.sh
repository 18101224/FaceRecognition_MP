#!/usr/bin/env bash
set -euo pipefail

# 사용법:
#   1) conda activate <env>  # 환경 활성화 먼저
#   2) ./install_groupy.sh
#
# 동작:
#   - 임시 작업 폴더에 shallow clone
#   - pip로 chainer, nose 설치 (현재 활성화된 환경에)
#   - setup.py install
#   - 임시 폴더 삭제 (성공/실패 시 모두)

REPO_URL="https://github.com/adambielski/GrouPy.git"

# 임시 작업 디렉터리 생성
WORKDIR="$(mktemp -d)"
cleanup() {
  # 작업 디렉터리 정리
  rm -rf "$WORKDIR"
  echo "Temporary directory cleaned up."
}
trap cleanup EXIT

echo "Cloning into $WORKDIR ..."
git clone --depth 1 "$REPO_URL" "$WORKDIR/GrouPy"

cd "$WORKDIR/GrouPy"

# pip 최신화 (선택적, 안정성을 위해)
python -m pip install --upgrade pip wheel setuptools

# 필요한 패키지 설치
echo "Installing dependencies: chainer and nose"
python -m pip install --no-cache-dir chainer nose

# 패키지 설치
echo "Installing GrouPy via setup.py"
python setup.py install

echo "Installation complete in the active environment."
