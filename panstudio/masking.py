from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def load_play_area_mask(path: Path, width: int, height: int) -> np.ndarray:
    """Load a direct binary mask or turn a closed red outline into one."""
    annotated = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if annotated is None:
        raise ValueError(f"Could not read play-area image: {path}")
    if annotated.ndim == 2:
        return cv2.resize(
            (annotated > 127).astype(np.uint8) * 255,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    if annotated.shape[2] == 4:
        annotated = annotated[:, :, :3]
    annotated = cv2.resize(annotated, (width, height), interpolation=cv2.INTER_AREA)
    blue, green, red = cv2.split(annotated)
    red_ink = (red > 155) & (red.astype(np.int16) - green.astype(np.int16) > 55) & (
        red.astype(np.int16) - blue.astype(np.int16) > 55
    )
    candidates = []
    ink = (red_ink.astype(np.uint8) * 255)
    ink = cv2.morphologyEx(
        ink,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
        iterations=2,
    )
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        _, _, w, h = cv2.boundingRect(contour)
        area = abs(cv2.contourArea(contour))
        if w > width * 0.65 and h > height * 0.18 and area > width * height * 0.12:
            candidates.append(contour)
    if not candidates:
        raise ValueError("No large closed red play-area outline was found")
    outline = max(candidates, key=cv2.contourArea)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [outline], -1, 255, cv2.FILLED)
    return mask


def save_mask_preview(mask: np.ndarray, clean_frame: np.ndarray, path: Path):
    preview = clean_frame.copy()
    excluded = mask == 0
    tint = np.zeros_like(preview)
    tint[:, :] = (20, 20, 210)
    preview[excluded] = cv2.addWeighted(preview[excluded], 0.38, tint[excluded], 0.62, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(preview, contours, -1, (40, 230, 40), 14)
    cv2.imwrite(str(path), preview)
