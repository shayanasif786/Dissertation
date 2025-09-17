
import os
import tables
import numpy as np
import nibabel as nib
from tqdm import tqdm
from glob import glob
from typing import Tuple, List, Optional, Dict
from config import cfg

# ---------- Helpers ----------

def _modality_globs(brain_dir: str):
    return {
        "flair": glob(os.path.join(brain_dir, "*-t2f.nii.gz")),
        "t1":    glob(os.path.join(brain_dir, "*-t1n.nii.gz")),
        "t1ce":  glob(os.path.join(brain_dir, "*-t1c.nii.gz")),
        "t2":    glob(os.path.join(brain_dir, "*-t2w.nii.gz")),
        "seg":   glob(os.path.join(brain_dir, "*-seg.nii.gz")),
    }

def _assert_all_modalities(mods, require_seg: bool=True):
    need = ["flair", "t1", "t1ce", "t2"]
    for k in need:
        if len(mods[k]) != 1:
            raise FileNotFoundError(f"Missing or multiple {k} in: {mods}")
    if require_seg and len(mods["seg"]) != 1:
        raise FileNotFoundError("Missing segmentation mask (seg).")

def read_brain(
    brain_dir: str,
    mode: str = "train",
    x0: int = 56, x1: int = 200,
    y0: int = 56, y1: int = 200,
    z0: int = 13, z1: int = 141,
):
    """
    Read all modalities (+ seg) and crop to bbox, returning [X,Y,Z,C] (C=5 with seg).
    """
    brain_dir = os.path.normpath(brain_dir)
    mods = _modality_globs(brain_dir)
    _assert_all_modalities(mods, require_seg=True)

    modality_paths = [mods["flair"][0], mods["t1"][0], mods["t1ce"][0], mods["t2"][0], mods["seg"][0]]

    arrays = []
    affine = None
    for p in modality_paths:
        ni = nib.load(p)
        arr = np.asarray(ni.dataobj)
        arrays.append(arr)
        affine = ni.affine

    vol = np.array(arrays)
    vol = np.rint(vol).astype(np.int16)
    vol = vol[:, x0:x1, y0:y1, z0:z1]     # crop
    vol = np.transpose(vol, (1, 2, 3, 0))   # (C, X, Y, Z) -> (X, Y, Z, C)              
    name = os.path.basename(brain_dir)
    bbox = (x0, x1, y0, y1, z0, z1)
    return vol, affine, name, bbox

def create_table(
    dataset_glob: str,
    table_data_shape: Tuple[int, int, int],
    save_dir: str,
    crop_coordinates: Dict[str, int],
    data_channels: int,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 100,
):
    """
    Build one HDF5 with labeled data only (BraTS-GoAT 2024), then
    create 3-way split indices: train_idx.npy, val_idx.npy, test_idx.npy.
    Stored nodes:
      - /data:  uint16 [N,X,Y,Z,4]
      - /truth:  uint8 [N,X,Y,Z]
      - /affine: float32 [N,4,4]
      - /bbox:   int16 [N,6]
      - /brain_names: list[str]
    """
    os.makedirs(save_dir, exist_ok=True)
    brains = sorted(glob(dataset_glob))

    h5 = tables.open_file(os.path.join(save_dir, "data.hdf5"), mode="w")
    filters = None

    data_shape   = tuple([0] + list(table_data_shape) + [data_channels])
    truth_shape  = tuple([0] + list(table_data_shape))
    affine_shape = (0, 4, 4)
    bbox_shape   = (0, 6)

    X = h5.create_earray(h5.root, "data",   tables.UInt16Atom(), shape=data_shape,   filters=filters, expectedrows=len(brains))
    Y = h5.create_earray(h5.root, "truth",  tables.UInt8Atom(),  shape=truth_shape,  filters=filters, expectedrows=len(brains))
    A = h5.create_earray(h5.root, "affine", tables.Float32Atom(),shape=affine_shape, filters=filters, expectedrows=len(brains))
    B = h5.create_earray(h5.root, "bbox",   tables.Int16Atom(),  shape=bbox_shape,   filters=filters, expectedrows=len(brains))

    names: List[str] = []
    for d in tqdm(brains, desc="ingest"):
        vol, aff, name, bbox = read_brain(d, mode="train", **crop_coordinates)
        brain = vol[..., :4].astype(np.uint16, copy=False)
        gt    = vol[..., -1].astype(np.uint8,  copy=False)  # labels {0,1,2,3} in BraTS-GoAT
        assert tuple(brain.shape[:3]) == tuple(table_data_shape), f"shape mismatch {brain.shape[:3]} vs {table_data_shape}"

        X.append(brain[np.newaxis, ...])
        Y.append(gt[np.newaxis, ...])
        A.append(aff[np.newaxis, ...])
        B.append(np.array(bbox, dtype=np.int16)[np.newaxis, ...])
        names.append(name)

    h5.create_array(h5.root, "brain_names", obj=names)
    h5.set_node_attr(h5.root, "norm", "none")  # normalization happens in generator
    h5.close()

    # ----- 3-way split -----
    rng = np.random.RandomState(seed)
    idx = np.arange(len(names))
    rng.shuffle(idx)

    n = len(idx)
    n_val  = int(round(val_frac * n))
    n_test = int(round(test_frac * n))
    n_train = n - n_val - n_test

    train_idx = np.sort(idx[:n_train])
    val_idx   = np.sort(idx[n_train:n_train+n_val])
    test_idx  = np.sort(idx[n_train+n_val:])

    np.save(os.path.join(save_dir, "train_idx.npy"), train_idx)
    np.save(os.path.join(save_dir, "val_idx.npy"),   val_idx)
    np.save(os.path.join(save_dir, "test_idx.npy"),  test_idx)
    print(f"Split => train:{len(train_idx)}  val:{len(val_idx)}  test:{len(test_idx)}")

if __name__ == "__main__":
    create_table(
        cfg["data_dir"],              # e.g., "/path/BraTS-GoAT-2024/train/*"
        cfg["table_data_shape"],      # (X,Y,Z) after crop
        cfg["save_data_dir"],         # folder path, ends with '/'
        cfg["crop_coord"],            # dict with x0,x1,y0,y1,z0,z1
        cfg["data_channels"],         # 4
        cfg.get("val_frac", 0.15),
        cfg.get("test_frac", 0.15),
        cfg.get("seed", 100),
    )
