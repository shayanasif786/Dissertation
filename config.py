
# config.py — aligned with prepare_data (2).py

import os

cfg = dict()

# ----- Crop (divisible by 8 for levels=3) -----
cfg['crop_coord'] = {
    'x0': 56, 'x1': 200,   # X = 144
    'y0': 56, 'y1': 200,   # Y = 144
    'z0': 13, 'z1': 141    # Z = 128
}

# ----- Dataset shape (X, Y, Z) after crop -----
cfg['table_data_shape'] = (
    cfg['crop_coord']['x1'] - cfg['crop_coord']['x0'],  # 144
    cfg['crop_coord']['y1'] - cfg['crop_coord']['y0'],  # 144
    cfg['crop_coord']['z1'] - cfg['crop_coord']['z0'],  # 128
)

# ----- Paths -----
# Folder with patient subfolders (labeled GoAT training data)
cfg['data_dir'] = "/content/drive/MyDrive/Dissertation/BraTS-GoAT/MICCAI2024-BraTS-GoAT-TrainingData-With-GroundTruth/*"
# Where to save HDF5 + split indices
cfg['save_data_dir'] = r"/content/drive/MyDrive/Dissertation/Project1/data"
cfg['save_dir'] = r"/content/drive/MyDrive/Dissertation/Project1/save"

# cfg['load_model_dir'] = r"/content/drive/MyDrive/Dissertation/Project1/save/axial_run/best_model.keras"

# ----- Split fractions (used by prepare_data (2).py) -----
cfg['val_frac'] = 0.15
cfg['test_frac'] = 0.15
cfg['seed'] = 100

# ----- HDF5 & Index files (use os.path.join for safety) -----
cfg['hdf5_dir']  = os.path.join(cfg['save_data_dir'], "data.hdf5")
cfg['train_idx'] = os.path.join(cfg['save_data_dir'], "train_idx.npy")
cfg['val_idx']   = os.path.join(cfg['save_data_dir'], "val_idx.npy")
cfg['test_idx']  = os.path.join(cfg['save_data_dir'], "test_idx.npy")

# ----- Data specifics -----
cfg['data_channels'] = 4
cfg['view'] = 'axial'  # used by data generator, not by prepare_data

# ----- Loader / Augmentation -----
cfg['batch_size'] = 8
cfg['val_batch_size'] = 16
cfg['hor_flip'] = True
cfg['ver_flip'] = True
cfg['rotation_range'] = 7     # degrees
cfg['zoom_range'] = 0.05      # ±5%

# ----- Training (tune for less overfitting) -----
cfg['epochs'] = 100
cfg['lr'] = 0.0005

# ----- Parallelism (keep off to avoid HDF5 lock issues) -----
cfg['multiprocessing'] = False
cfg['workers'] = 1

# ----- Model -----
cfg['modified_unet'] = True
cfg['levels'] = 3
cfg['start_chs'] = 32    # smaller width than 64 to reduce overfitting


# ----- Disable old k-fold logic (BraTS 2018-style) -----
cfg['k_fold'] = None
