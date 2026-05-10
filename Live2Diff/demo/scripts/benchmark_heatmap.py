#!/usr/bin/env python3
"""
从稳定性测试导出的 JSON 生成二维热力图：两轴为参数，颜色表示 score（整体 min-max 归一化，蓝=小，红=大）。

用法:
  python benchmark_heatmap.py /path/to/benchmark.json -o heatmap.png
  python benchmark_heatmap.py data.json --x-param depth_scale --y-param softedge_scale
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ImportError as e:
    print("需要 matplotlib: pip install matplotlib", file=sys.stderr)
    raise SystemExit(1) from e


def _float_key(x: float, tol: float = 1e-6) -> float:
    return round(x / tol) * tol


def _match_axis_value(value: float, axis_values: List[float], tol: float = 1e-5) -> Optional[int]:
    for i, v in enumerate(axis_values):
        if abs(value - v) <= tol * max(1.0, abs(v)):
            return i
    return None


def load_benchmark_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def pick_xy_params(
    data: Dict[str, Any],
    x_param: Optional[str],
    y_param: Optional[str],
) -> Tuple[str, str, List[float], List[float]]:
    space = data.get("search_space") or []
    if len(space) < 2:
        raise ValueError("JSON 中 search_space 至少需要 2 个参数")

    if x_param and y_param:
        x_spec = next((s for s in space if s.get("param_id") == x_param), None)
        y_spec = next((s for s in space if s.get("param_id") == y_param), None)
        if not x_spec or not y_spec:
            raise ValueError("指定的 --x-param / --y-param 在 search_space 中未找到")
    else:
        x_spec, y_spec = space[0], space[1]

    xv = sorted({float(v) for v in (x_spec.get("values") or [])})
    yv = sorted({float(v) for v in (y_spec.get("values") or [])})
    if not xv or not yv:
        raise ValueError("search_space 中缺少 values")

    return str(x_spec["param_id"]), str(y_spec["param_id"]), xv, yv


def build_grid(
    data: Dict[str, Any],
    x_param: str,
    y_param: str,
    x_vals: List[float],
    y_vals: List[float],
) -> Tuple[Any, List[float], List[float]]:
    nx, ny = len(x_vals), len(y_vals)
    grid = [[math.nan for _ in range(nx)] for _ in range(ny)]

    for row in data.get("results") or []:
        params = row.get("params") or {}
        if x_param not in params or y_param not in params:
            continue
        score = row.get("score")
        if score is None or not isinstance(score, (int, float)) or not math.isfinite(score):
            continue
        xi = _match_axis_value(float(params[x_param]), x_vals)
        yi = _match_axis_value(float(params[y_param]), y_vals)
        if xi is None or yi is None:
            continue
        grid[yi][xi] = float(score)

    return grid, x_vals, y_vals


def blue_red_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("blue_small_red_large", ["#0000ff", "#ff0000"])


def main() -> None:
    ap = argparse.ArgumentParser(description="稳定性测试 JSON → 二维热力图")
    ap.add_argument("json_path", type=Path, help="benchmark 导出的 .json 路径")
    ap.add_argument("-o", "--output", type=Path, default=None, help="输出图片路径，默认与 json 同名 .png")
    ap.add_argument("--x-param", type=str, default=None, help="横轴参数 id（默认 search_space 第 1 个）")
    ap.add_argument("--y-param", type=str, default=None, help="纵轴参数 id（默认 search_space 第 2 个）")
    ap.add_argument("--dpi", type=int, default=150, help="输出 DPI")
    ap.add_argument("--title", type=str, default=None, help="图标题")
    args = ap.parse_args()

    if not args.json_path.is_file():
        raise SystemExit(f"文件不存在: {args.json_path}")

    data = load_benchmark_json(args.json_path)
    x_param, y_param, x_vals, y_vals = pick_xy_params(data, args.x_param, args.y_param)
    grid, x_vals, y_vals = build_grid(data, x_param, y_param, x_vals, y_vals)

    flat = [c for row in grid for c in row if not math.isnan(c)]
    if not flat:
        raise SystemExit("没有可用的数值 score，无法绘图")

    vmin, vmax = min(flat), max(flat)
    if vmin == vmax:
        vmax = vmin + 1e-9

    cmap = blue_red_cmap()
    out = args.output or args.json_path.with_suffix(".png")

    fig, ax = plt.subplots(figsize=(max(6, len(x_vals) * 0.45), max(5, len(y_vals) * 0.45)), dpi=args.dpi)
    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([f"{v:g}" for v in x_vals])
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([f"{v:g}" for v in y_vals])

    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    if args.title:
        ax.set_title(args.title)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for yi in range(len(y_vals)):
        for xi in range(len(x_vals)):
            v = grid[yi][xi]
            if math.isnan(v):
                continue
            t = (v - vmin) / (vmax - vmin)
            ax.text(
                xi,
                yi,
                f"{v:.3g}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if t > 0.55 else "black",
            )

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"已写入: {out}")


if __name__ == "__main__":
    main()
