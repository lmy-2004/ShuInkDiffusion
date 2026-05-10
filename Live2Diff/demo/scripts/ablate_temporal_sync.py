#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from temporal_stability import compute_flow_warp_errors, compute_optical_flow, stability_score

try:
    import cv2
except ImportError as e:
    raise SystemExit("需要 opencv-python") from e


def read_video(path: Path, max_frames: Optional[int]) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {path}")
    frames: List[np.ndarray] = []
    while max_frames is None or len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def score_pair(src: List[np.ndarray], out: List[np.ndarray], flow_max_side: int, flow_backend: str) -> Dict[str, Any]:
    n = min(len(src), len(out))
    if n < 2:
        return {"frames": n, "e_src": None, "e_out": None, "score_src": None, "score_out": None, "delta": None}
    e_srcs: List[float] = []
    e_outs: List[float] = []
    for i in range(1, n):
        flow = compute_optical_flow(src[i - 1], src[i], flow_max_side=flow_max_side, backend=flow_backend)
        pair = compute_flow_warp_errors(src[i - 1], src[i], out[i - 1], out[i], flow_max_side, flow)
        if pair is None:
            continue
        e_src, e_out = pair
        e_srcs.append(e_src)
        e_outs.append(e_out)
    if not e_outs:
        return {"frames": n, "e_src": None, "e_out": None, "score_src": None, "score_out": None, "delta": None}
    e_src_m = mean(e_srcs)
    e_out_m = mean(e_outs)
    s_src = stability_score(e_src_m)
    s_out = stability_score(e_out_m)
    return {
        "frames": n,
        "pairs": len(e_outs),
        "e_src": round(e_src_m, 6),
        "e_out": round(e_out_m, 6),
        "score_src": round(s_src, 4),
        "score_out": round(s_out, 4),
        "delta": round(s_out - s_src, 4),
    }


def eval_case(case: Dict[str, Any], max_frames: Optional[int], flow_max_side: int, flow_backend: str) -> Dict[str, Any]:
    src = read_video(Path(case["source"]), max_frames)
    result: Dict[str, Any] = {"name": case.get("name", "case"), "category": case.get("category", "unknown")}
    for key in ("baseline", "sync"):
        if key in case and case[key]:
            result[key] = score_pair(src, read_video(Path(case[key]), max_frames), flow_max_side, flow_backend)
    if "baseline" in result and "sync" in result:
        b = result["baseline"].get("score_out")
        s = result["sync"].get("score_out")
        result["sync_gain"] = None if b is None or s is None else round(s - b, 4)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="三类素材 A/B 时序稳定性汇总")
    ap.add_argument("manifest", type=Path, help="JSON: cases=[{name,category,source,baseline,sync}]")
    ap.add_argument("-o", "--output", type=Path, default=Path("temporal_sync_ablation.json"))
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--flow-max-side", type=int, default=256)
    ap.add_argument("--flow-backend", type=str, default="farneback")
    args = ap.parse_args()

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    results = [eval_case(c, args.max_frames, args.flow_max_side, args.flow_backend) for c in data.get("cases", [])]
    out = {
        "flow_backend": args.flow_backend,
        "flow_max_side": args.flow_max_side,
        "results": results,
    }
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入: {args.output}")


if __name__ == "__main__":
    main()
