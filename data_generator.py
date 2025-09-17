# data_generator.py
# Compatible with HDF5 built by your prepare_data (2).py:
#   /data  -> float32 [N, X, Y, Z, C]
#   /truth -> uint8   [N, X, Y, Z]
#
# Views:
#   axial    : Slices along Z; upright display => (H=Y, W=X)
#   coronal  : Slices along Y; upright display => (H=Z, W=X)
#   sagittal : Slices along X; upright display => (H=Z, W=Y)
#
# Overfitting controls baked in:
#   - Foreground-aware sampling (cap background fraction per batch)
#   - Per-slice percentile clip (1–99%) + z-score (per channel)
#   - Light augments (train only): flips, small rotation, small zoom
#
# Notes:
#   - Keep cfg['multiprocessing']=False and workers=1 to avoid HDF5 locks.
#   - If scipy is unavailable, rotation/zoom are disabled gracefully.

from __future__ import annotations
import math
import random
from typing import List, Tuple, Optional

import numpy as np
import tensorflow as tf
import h5py

try:
    from scipy.ndimage import rotate as nd_rotate, zoom as nd_zoom
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from tensorflow.keras.utils import Sequence, to_categorical

class CustomDataGenerator(Sequence):

    """
    Keras-like generator yielding (X, y) batches from an HDF5:
      - Patient-independent splits via provided brain_idx (train/val/test indices).
      - Foreground-aware slice sampling during training.
      - Augmentations only in training.
    """

    def __init__(
        self,
        hdf5_file: str,
        brain_idx: List[int],
        batch_size: int = 8,
        view: str = "axial",              # 'axial' | 'coronal' | 'sagittal'
        mode: str = "train",              # 'train' | 'validation' | 'test'
        horizontal_flip: bool = True,
        vertical_flip: bool = True,
        rotation_range: float = 10.0,     # degrees (small)
        zoom_range: float = 0.10,         # fraction (e.g., 0.10 => ±10%)
        bg_max_frac: float = 0.15,        # at most 40% background-only slices per batch
        shuffle: bool = True,
        seed: Optional[int] = 42,
        steps_per_epoch: Optional[int] = None,   # if None, computed from pool sizes
    ):
        assert 0.0 <= bg_max_frac < 1.0, "bg_max_frac must be in [0,1)"
        self.h5_path = hdf5_file
        self.h5 = hdf5_file
        self.data_storage = self.h5["data"]   # [N, X, Y, Z, C]
        
        # Axes for slicing
        if view == "axial":
           view_axes_img = (2, 1, 0, 3)  # slices along Z → (Y,X,C)
        elif view == "coronal":
             view_axes_img = (1, 2, 0, 3)
        elif view == "sagittal":
             view_axes_img = (0, 2, 1, 3)
        else:
             raise ValueError("Invalid view")

        # Save data shape (S, H, W, C)
        self.data_shape = tuple(np.array(self.data_storage.shape[1:])[list(view_axes_img)])
        
        self.truth_storage = self.h5["truth"] # [N, X, Y, Z]

        self.N = self.data_storage.shape[0]
        self.brain_idx = np.array(brain_idx, dtype=int).tolist()
        self.batch_size = int(batch_size)
        self.view = view.lower()
        self.mode = mode.lower()
        self.h_flip = bool(horizontal_flip)
        self.v_flip = bool(vertical_flip)
        self.rot_deg = float(rotation_range)
        self.zoom_rng = float(zoom_range)
        self.bg_max_frac = float(bg_max_frac)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        # Determine slicing axis (S axis in original [X,Y,Z])
        if self.view == "axial":
            self.slice_axis = 2  # Z
            # upright: (H=Y, W=X) -> transpose (X,Y,C) -> (Y,X,C)
            self._orient_axes_img = (1, 0, 2)  # for per-slice arrays
            self._orient_axes_lbl = (1, 0)
        elif self.view == "coronal":
            self.slice_axis = 1  # Y
            # upright: (H=Z, W=X) -> from (X,Z,C) -> (Z,X,C)
            self._orient_axes_img = (1, 0, 2)
            self._orient_axes_lbl = (1, 0)
        elif self.view == "sagittal":
            self.slice_axis = 0  # X
            # upright: (H=Z, W=Y) -> from (Y,Z,C) -> (Z,Y,C)
            self._orient_axes_img = (1, 0, 2)
            self._orient_axes_lbl = (1, 0)
        else:
            raise ValueError(f"Unknown view={view}. Use 'axial'|'coronal'|'sagittal'.")

        # Shapes
        _, X, Y, Z, C = self._shape_5d(self.data_storage.shape)
        S = [X, Y, Z][self.slice_axis]
        self.channels = C
        self._S = S  # slices per brain along chosen view

        # Build slice pools (tumor/background) for training; full sequential list otherwise
        if self.mode == "train":
            self.fg_pool, self.bg_pool = self._build_slice_pools()
            # Estimate steps if not provided: cap bg to maintain bg_max_frac
            fg = len(self.fg_pool)
            bg_cap = int(fg * (self.bg_max_frac / max(1e-6, (1.0 - self.bg_max_frac))))
            total = fg + min(len(self.bg_pool), bg_cap)
            self._steps = steps_per_epoch if steps_per_epoch is not None else max(1, math.ceil(total / self.batch_size))
        else:
            self.seq_list = self._build_sequential_list()
            self._steps = max(1, math.ceil(len(self.seq_list) / self.batch_size))

        # Log a concise summary
        H, W = self._infer_hw()
        print(
            f"[DataGen] mode={self.mode} | brains={len(self.brain_idx)} "
            f"| slices/brain={self._S} | slice_shape={H}x{W}x{self.channels} | view={self.view} "
            f"| steps/epoch={self._steps}"
        )

    # ---------- Public API ----------

    def __len__(self) -> int:
        return self._steps

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
    

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.mode == "train":
            return self._get_batch_train()
        else:
            return self._get_batch_eval(idx)

    def on_epoch_end(self):
        if self.mode == "train" and self.shuffle:
            self.rng.shuffle(self.fg_pool)
            self.rng.shuffle(self.bg_pool)
        elif self.mode != "train" and self.shuffle:
            self.rng.shuffle(self.seq_list)

    def close(self):
        try:
            self.h5.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

    # ---------- Internal: pools & lists ----------

    def _build_slice_pools(self) -> Tuple[List[Tuple[int,int]], List[Tuple[int,int]]]:
        """Return (fg_pool, bg_pool) as lists of (brain_id, slice_id)."""
        fg, bg = [], []
        for b in self.brain_idx:
            lbl = self.truth_storage[b]  # [X,Y,Z]
            # any tumor > 0 in slice along selected axis
            tumor_mask_per_slice = self._has_tumor_per_slice(lbl, axis=self.slice_axis)
            all_slices = np.arange(self._S)
            fg_slices = all_slices[tumor_mask_per_slice]
            bg_slices = all_slices[~tumor_mask_per_slice]
            fg.extend([(b, int(s)) for s in fg_slices])
            bg.extend([(b, int(s)) for s in bg_slices])

        if self.shuffle:
            self.rng.shuffle(fg)
            self.rng.shuffle(bg)
        return fg, bg

    def _build_sequential_list(self) -> List[Tuple[int,int]]:
        """Sequential coverage across all slices and brains (for val/test)."""
        seq = []
        for b in (self.brain_idx if not self.shuffle else self.rng.permutation(self.brain_idx)):
            for s in range(self._S):
                seq.append((b, s))
        return seq

    @staticmethod
    def _has_tumor_per_slice(lbl_3d: np.ndarray, axis: int) -> np.ndarray:
        """True if any voxel > 0 in the slice along given axis."""
        return np.any(lbl_3d > 0, axis=tuple(i for i in range(3) if i != axis))

    # ---------- Internal: batch assembly ----------

    def _get_batch_train(self) -> Tuple[np.ndarray, np.ndarray]:
        # Determine counts
        bg_quota = int(round(self.bg_max_frac * self.batch_size))
        fg_quota = self.batch_size - bg_quota

        # Sample
        fg_samples = self._draw(self.fg_pool, fg_quota)
        bg_samples = self._draw(self.bg_pool, bg_quota)

        samples = fg_samples + bg_samples
        if self.shuffle:
            self.rng.shuffle(samples)

        Xs, Ys = [], []
        for b, s in samples:
            x, y = self._read_slice(b, s)          # shapes: x[H,W,C], y[H,W]
            x = self._normalize_slice(x)           # per-channel
            x, y = self._maybe_augment(x, y)       # train-only aug
            Xs.append(x)
            Ys.append(y)
        X = np.stack(Xs, axis=0).astype(np.float32)
                                                                             
        Y = to_categorical(Ys, 4).astype(np.float32) 

        return X, Y


       


    def _get_batch_eval(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        start = idx * self.batch_size
        end = min(len(self.seq_list), start + self.batch_size)
        batch = self.seq_list[start:end]

        Xs, Ys = [], []
        for b, s in batch:
            x, y = self._read_slice(b, s)          # shapes: x[H,W,C], y[H,W]
            x = self._normalize_slice(x)
            Xs.append(x)
            Ys.append(y)
        X = np.stack(Xs, axis=0).astype(np.float32)
  
        Y = to_categorical(Ys, 4).astype(np.float32)  # 


        return X, Y


    def _draw(self, pool: List[Tuple[int,int]], k: int) -> List[Tuple[int,int]]:
        if k <= 0 or not pool:
            return []
        # If pool has fewer than k, sample with replacement to keep batch size stable
        if len(pool) >= k:
            idxs = self.rng.choice(len(pool), size=k, replace=False)
        else:
            idxs = self.rng.choice(len(pool), size=k, replace=True)
        return [pool[i] for i in idxs]

    # ---------- Internal: IO & orientation ----------

    def _read_slice(self, brain_id: int, s: int) -> Tuple[np.ndarray, np.ndarray]:
        """Read one slice (x,y) with upright orientation for the selected view."""
        vol = self.data_storage[brain_id]   # [X,Y,Z,C]
        lbl = self.truth_storage[brain_id]  # [X,Y,Z]

        if self.view == "axial":
            # take along Z
            x = vol[:, :, s, :]            # [X,Y,C]
            y = lbl[:, :, s]               # [X,Y]
        elif self.view == "coronal":
            # take along Y
            x = vol[:, s, :, :]            # [X,Z,C]
            y = lbl[:, s, :]               # [X,Z]
        else:  # sagittal
            # take along X
            x = vol[s, :, :, :]            # [Y,Z,C]
            y = lbl[s, :, :]               # [Y,Z]

        # Upright orientation (see __init__ comments)
        x = np.transpose(x, self._orient_axes_img)  # -> [H,W,C]
        y = np.transpose(y, self._orient_axes_lbl)  # -> [H,W]
        return x, y

    def _infer_hw(self) -> Tuple[int, int]:
        # Peek first brain to get H,W after orientation
        b0 = self.brain_idx[0]
        x0, _ = self._read_slice(b0, 0)
        return x0.shape[0], x0.shape[1]

    # ---------- Internal: normalization & augmentation ----------

    def _normalize_slice(self, x: np.ndarray) -> np.ndarray:
        """Per-slice, per-channel: clip to [p1,p99] (nonzero), then z-score."""
        x = x.astype(np.float32)
        H, W, C = x.shape
        out = np.empty_like(x, dtype=np.float32)

        for c in range(C):
            chan = x[..., c]
            nz = chan[np.nonzero(chan)]
            if nz.size < 16:
                # fall back to simple scaling if slice is near-empty
                mean, std = 0.0, 1.0
                p1, p99 = np.percentile(chan, 1), np.percentile(chan, 99)
            else:
                p1, p99 = np.percentile(nz, 1), np.percentile(nz, 99)
                chan = np.clip(chan, p1, p99)
                nz = chan[np.nonzero(chan)]
                mean = float(np.mean(nz)) if nz.size > 0 else float(np.mean(chan))
                std = float(np.std(nz)) if nz.size > 1 else 1.0

            if std == 0.0:
                std = 1.0
            out[..., c] = (chan - mean) / std

        return out

    def _maybe_augment(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Train-time only, light & safe augmentations."""
        if self.mode != "train":
            return x, y

        # Random flips
        if self.h_flip and self.rng.random() < 0.5:
            x = np.flip(x, axis=1)  # horizontal (W)
            y = np.flip(y, axis=1)
        if self.v_flip and self.rng.random() < 0.5:
            x = np.flip(x, axis=0)  # vertical (H)
            y = np.flip(y, axis=0)

        # Rotation
        if _HAS_SCIPY and self.rot_deg > 0:
            deg = self.rng.uniform(-self.rot_deg, self.rot_deg)
            # order=1 for image, order=0 for labels; preserve shape
            x = np.stack(
                [nd_rotate(x[..., c], deg, reshape=False, order=1, mode="nearest") for c in range(x.shape[-1])],
                axis=-1
            )
            y = nd_rotate(y, deg, reshape=False, order=0, mode="nearest")

        # Zoom
        if _HAS_SCIPY and self.zoom_rng > 0:
            zf = float(self.rng.uniform(1.0 - self.zoom_rng, 1.0 + self.zoom_rng))
            x = self._zoom_keep_size(x, zf, order=1)
            y = self._zoom_keep_size(y, zf, order=0)

        return x, y

    def _zoom_keep_size(self, arr: np.ndarray, zf: float, order: int) -> np.ndarray:
        """Zoom 2D (or 2D+C) and center-crop/pad back to original size."""
        H, W = arr.shape[:2]
        if arr.ndim == 3:
            # zoom each channel equally
            z = np.stack([nd_zoom(arr[..., c], (zf, zf), order=order, mode="nearest") for c in range(arr.shape[-1])], axis=-1)
        else:
            z = nd_zoom(arr, (zf, zf), order=order, mode="nearest")

        h, w = z.shape[:2]
        # center crop/pad to (H, W)
        if h >= H:
            top = (h - H) // 2
            z = z[top:top + H, ...]
        else:
            pad = (H - h)
            top = pad // 2
            bot = pad - top
            pad_width = ((top, bot), (0, 0)) if arr.ndim == 2 else ((top, bot), (0, 0), (0, 0))
            z = np.pad(z, pad_width, mode='constant', constant_values=0)

        if w >= W:
            left = (w - W) // 2
            z = z[:, left:left + W, ...] if z.ndim == 3 else z[:, left:left + W]
        else:
            pad = (W - w)
            left = pad // 2
            right = pad - left
            pad_width = ((0, 0), (left, right)) if arr.ndim == 2 else ((0, 0), (left, right), (0, 0))
            z = np.pad(z, pad_width, mode='constant', constant_values=0)

        return z

    # ---------- Helpers ----------

    @staticmethod
    def _shape_5d(shape):
        assert len(shape) == 5, f"expected 5D shape [N,X,Y,Z,C], got {shape}"
        return shape

