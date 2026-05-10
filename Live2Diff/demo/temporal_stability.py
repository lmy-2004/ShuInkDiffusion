from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

STABILITY_K = 50.0
SUPPORTED_FLOW_BACKENDS = {"farneback", "raft", "sea-raft"}


def normalize_flow_backend(backend: str = "farneback") -> str:
    backend = str(backend).lower().replace("_", "-")
    return backend if backend in SUPPORTED_FLOW_BACKENDS else "farneback"


class FlowStats(NamedTuple):
    flow: np.ndarray
    map_x: np.ndarray
    map_y: np.ndarray
    e_src: float
    mean_flow_mag: float
    backend: str = "farneback"


def stability_score(e: float) -> float:
    return 100.0 / (1.0 + STABILITY_K * e)


def _validate_frame_pair(prev_in_rgb: np.ndarray, curr_in_rgb: np.ndarray) -> bool:
    if cv2 is None:
        return False
    if prev_in_rgb.shape != curr_in_rgb.shape:
        return False
    h, w = curr_in_rgb.shape[:2]
    return h >= 4 and w >= 4


def warp_with_maps(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> Optional[np.ndarray]:
    if cv2 is None:
        return None
    if image.ndim == 2:
        return cv2.remap(
            image.astype(np.float32),
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
    if image.ndim != 3:
        return None
    warped = np.empty_like(image, dtype=np.float32)
    for c in range(image.shape[2]):
        warped[:, :, c] = cv2.remap(
            image[:, :, c].astype(np.float32),
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
    return warped


def compute_optical_flow(
    prev_in_rgb: np.ndarray,
    curr_in_rgb: np.ndarray,
    flow_max_side: int = 256,
    backend: str = "farneback",
) -> Optional[FlowStats]:
    if not _validate_frame_pair(prev_in_rgb, curr_in_rgb):
        return None
    backend = normalize_flow_backend(backend)

    h, w = curr_in_rgb.shape[:2]
    prev_g = cv2.cvtColor(prev_in_rgb, cv2.COLOR_RGB2GRAY)
    curr_g = cv2.cvtColor(curr_in_rgb, cv2.COLOR_RGB2GRAY)

    scale = min(flow_max_side / max(h, w), 1.0)
    wf = max(4, int(round(w * scale)))
    hf = max(4, int(round(h * scale)))
    if wf != w or hf != h:
        prev_gs = cv2.resize(prev_g, (wf, hf), interpolation=cv2.INTER_AREA)
        curr_gs = cv2.resize(curr_g, (wf, hf), interpolation=cv2.INTER_AREA)
    else:
        prev_gs, curr_gs = prev_g, curr_g

    flow_s = cv2.calcOpticalFlowFarneback(
        prev_gs, curr_gs, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    sx = w / float(wf)
    sy = h / float(hf)
    flow = cv2.resize(flow_s, (w, h), interpolation=cv2.INTER_LINEAR)
    flow[:, :, 0] *= sx
    flow[:, :, 1] *= sy

    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    map_x = grid_x + flow[:, :, 0]
    map_y = grid_y + flow[:, :, 1]

    warped_prev_g = warp_with_maps(prev_g, map_x, map_y)
    if warped_prev_g is None:
        return None

    e_src = float(
        np.mean(
            np.abs(curr_g.astype(np.float32) / 255.0 - warped_prev_g / 255.0)
        )
    )
    mean_flow_mag = float(np.mean(np.linalg.norm(flow, axis=2)))
    return FlowStats(
        flow=flow,
        map_x=map_x,
        map_y=map_y,
        e_src=e_src,
        mean_flow_mag=mean_flow_mag,
        backend="farneback",
    )


def compute_flow_warp_errors(
    prev_in_rgb: np.ndarray,
    curr_in_rgb: np.ndarray,
    prev_out_rgb: np.ndarray,
    curr_out_rgb: np.ndarray,
    flow_max_side: int = 256,
    flow_stats: Optional[FlowStats] = None,
    flow_backend: str = "farneback",
) -> Optional[Tuple[float, float]]:
    if flow_stats is None:
        flow_stats = compute_optical_flow(
            prev_in_rgb,
            curr_in_rgb,
            flow_max_side=flow_max_side,
            backend=flow_backend,
        )
    if flow_stats is None:
        return None
    if (
        prev_out_rgb.shape != curr_out_rgb.shape
        or curr_in_rgb.shape[:2] != curr_out_rgb.shape[:2]
    ):
        return None
    warped_out = warp_with_maps(prev_out_rgb, flow_stats.map_x, flow_stats.map_y)
    if warped_out is None:
        return None
    e_out = float(np.mean(np.abs(curr_out_rgb.astype(np.float32) / 255.0 - warped_out / 255.0)))
    return flow_stats.e_src, e_out
