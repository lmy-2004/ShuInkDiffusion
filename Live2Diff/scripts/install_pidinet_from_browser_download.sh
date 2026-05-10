#!/usr/bin/env bash
set -euo pipefail

# PiDiNet 权重：table5_pidinet.pth（controlnet_aux / lllyasviel/Annotators）
# 用法：运行后会在浏览器打开下载页；下载完成后把文件保存为 table5_pidinet.pth（默认进「下载」文件夹），脚本会轮询并装入 HF 缓存。

FILE="table5_pidinet.pth"
REPO_ID="lllyasviel/Annotators"
# hf-mirror 与官方同一 revision（见 https://hf-mirror.com/api/models/lllyasviel/Annotators/revision/main）
REVISION_SHA="982e7edaec38759d914a963c48c4726685de7d96"

URL_OFFICIAL="https://huggingface.co/${REPO_ID}/resolve/main/${FILE}"
URL_MIRROR="https://hf-mirror.com/${REPO_ID}/resolve/main/${FILE}?download=true"

DOWNLOADS="${HOME}/Downloads"
if [[ -n "${XDG_DOWNLOAD_DIR:-}" && -d "${XDG_DOWNLOAD_DIR}" ]]; then
  DOWNLOADS="${XDG_DOWNLOAD_DIR}"
fi

HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
HUB="${HF_HOME}/hub"
REPO_DIR="${HUB}/models--lllyasviel--Annotators"
SNAP="${REPO_DIR}/snapshots/${REVISION_SHA}"
REFS_DIR="${REPO_DIR}/refs"

echo "直接下载链接（官方）："
echo "  ${URL_OFFICIAL}"
echo "直接下载链接（镜像，国内常用）："
echo "  ${URL_MIRROR}"
echo ""
echo "安装目标（HF 缓存快照目录）："
echo "  ${SNAP}/${FILE}"
echo ""

if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "${URL_MIRROR}" 2>/dev/null || xdg-open "${URL_OFFICIAL}" 2>/dev/null || true
elif command -v open >/dev/null 2>&1; then
  open "${URL_MIRROR}" 2>/dev/null || open "${URL_OFFICIAL}" 2>/dev/null || true
fi

echo "请在浏览器中完成下载，文件名应为：${FILE}"
echo "正在监控目录: ${DOWNLOADS}"
echo ""

FOUND=""
for _ in $(seq 1 720); do
  if [[ -f "${DOWNLOADS}/${FILE}" ]]; then
    sz=$(stat -c%s "${DOWNLOADS}/${FILE}" 2>/dev/null || stat -f%z "${DOWNLOADS}/${FILE}" 2>/dev/null || echo 0)
    if [[ "${sz}" -gt 1000000 ]]; then
      FOUND="${DOWNLOADS}/${FILE}"
      break
    fi
  fi
  sleep 5
done

if [[ -z "${FOUND}" ]]; then
  echo "超时未在 ${DOWNLOADS} 找到大于 1MB 的 ${FILE}，请手动执行："
  echo "  mkdir -p \"${SNAP}\""
  echo "  cp /你的路径/${FILE} \"${SNAP}/${FILE}\""
  echo "  mkdir -p \"${REFS_DIR}\""
  echo "  echo -n '${REVISION_SHA}' > \"${REFS_DIR}/main\""
  exit 1
fi

mkdir -p "${SNAP}" "${REFS_DIR}"
cp -f "${FOUND}" "${SNAP}/${FILE}"
echo -n "${REVISION_SHA}" > "${REFS_DIR}/main"
chmod -R u+rwX "${REPO_DIR}" 2>/dev/null || true

echo ""
echo "已安装: ${SNAP}/${FILE}"
echo "请重启 Live2Diff / demo 进程；配置 softedge_mode: pidinet"
