
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------- Side-by-side Brain Tumor Prediction Mask with Ground Truth Mask ----------

# BraTS colormap (RGB only)
brats_rgb = {
    0: (0, 0, 0),      # Background - black
    1: (255, 255, 0),  # NCR - yellow
    2: (0, 255, 0),    # ED - green
    3: (255, 0, 0)     # ET - red
}

def mask_to_rgb(mask):
    """Convert [H,W] mask with {0,1,2,3} to RGB image."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for k, color in brats_rgb.items():
        rgb[mask == k] = color
    return rgb

def dice_iou_per_class(gt_mask, pred_mask, cls):
    y_true = (gt_mask == cls).astype(np.uint8)
    y_pred = (pred_mask == cls).astype(np.uint8)

    if np.sum(y_true) == 0:  # class absent in GT
        return None, None

    inter = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    dice = (2*inter + 1e-7) / (union + 1e-7)
    iou  = (inter + 1e-7) / (np.sum((y_true + y_pred) > 0) + 1e-7)
    return dice, iou

def normalize_like_generator(slice_2d):
    """Same per-slice z-score normalization with percentile clipping as in CustomDataGenerator."""
    out = np.empty_like(slice_2d, dtype=np.float32)
    for c in range(slice_2d.shape[-1]):
        chan = slice_2d[..., c].astype(np.float32)
        nz = chan[np.nonzero(chan)]
        if nz.size < 16:
            p1, p99 = np.percentile(chan, 1), np.percentile(chan, 99)
            mean, std = 0.0, 1.0
        else:
            p1, p99 = np.percentile(nz, 1), np.percentile(nz, 99)
            chan = np.clip(chan, p1, p99)
            nz = chan[np.nonzero(chan)]
            mean = float(np.mean(nz)) if nz.size > 0 else float(np.mean(chan))
            std = float(np.std(nz)) if nz.size > 1 else 1.0
        if std == 0: std = 1.0
        out[..., c] = (chan - mean) / std
    return out

def visualize_unique_patients(h5_file, test_idx, model, n_patients=3):
    """
    Visualize Flair, T1ce, Ground Truth, Prediction for unique patients.
    Each row = one patient, with Patient ID + metrics at top-left.
    """
    brain_names = [name.decode("utf-8") for name in h5_file["brain_names"][:]]
    shown = 0

    for pid in test_idx:
        vol = h5_file["data"][pid]   # [X,Y,Z,C]
        gt  = h5_file["truth"][pid]  # [X,Y,Z]
        patient_name = brain_names[pid]

        # Pick slice with max tumor
        tumor_sums = np.sum(gt > 0, axis=(0,1))
        slice_idx = np.argmax(tumor_sums)

        flair = vol[:,:,slice_idx,0]
        t1ce  = vol[:,:,slice_idx,2]
        gt_mask = gt[:,:,slice_idx]

        # Prepare input
        inp = vol[:,:,slice_idx,:]
        inp = normalize_like_generator(inp)  # use same norm as generator

        # Predict
        pred = model.predict(np.expand_dims(inp,0), verbose=0)
        pred_mask = np.argmax(pred[0], axis=-1)

        # Normalize MRI for display
        flair_disp = np.rot90((flair - flair.min())/(flair.max()-flair.min()+1e-8), k=3)
        t1ce_disp  = np.rot90((t1ce  - t1ce.min()) /(t1ce.max()-t1ce.min() +1e-8), k=3)

        # Convert masks to RGB
        gt_rgb   = np.rot90(mask_to_rgb(gt_mask), k=3)
        pred_rgb = np.rot90(mask_to_rgb(pred_mask), k=3)

        # ---- Metrics ----
        metrics_parts = []
        for cls,name in zip([1,2,3],["NCR","ED","ET"]):
            d,i = dice_iou_per_class(gt_mask, pred_mask, cls)
            if d is None:  # absent in GT
                metrics_parts.append(f"{name}: N/A")
            else:
                metrics_parts.append(f"{name}: Dice={d:.3f}, IoU={i:.3f}")

        metrics_line = " | ".join(metrics_parts)

        ## Adding tumor class legend
        legend_elements = [
            mpatches.Patch(color='yellow', label='NCR'),
            mpatches.Patch(color='green',  label='ED'),
            mpatches.Patch(color='red',    label='ET'),
            mpatches.Patch(color='black',  label='Background')
        ]
     


        # ---- Plot row ----
        fig, axs = plt.subplots(1,4, figsize=(12,4))
        axs[0].imshow(flair_disp, cmap="gray"); axs[0].set_title("FLAIR")
        axs[1].imshow(t1ce_disp, cmap="gray");  axs[1].set_title("T1ce")
        axs[2].imshow(gt_rgb); axs[2].set_title("Ground Truth")
        axs[3].imshow(pred_rgb); axs[3].set_title("Prediction")

        for a in axs: a.axis("off")

        # Add patient ID + metrics at top-left of the row
        title_text = f"Patient: {patient_name} | {metrics_line}"
        fig.text(0.01, 0.95, title_text, ha="left", va="top", fontsize=9)
        fig.legend(handles=legend_elements, loc='lower center', ncol=4, frameon=False)

        plt.tight_layout(pad=2.0)
        plt.show()

        shown += 1
        if shown >= n_patients:
            break




# ---------- Overlay Predicted Segmentation Mask using FLAIR Modality along with Error Map ----------

# BraTS colors
brats_colors = {
    1: (1, 1, 0),   # NCR - yellow
    2: (0, 1, 0),   # ED - green
    3: (1, 0, 0),   # ET - red
}

def overlay_mask_on_mri(mri_slice, mask, alpha=0.4):
    """Overlay segmentation mask on grayscale MRI slice."""
    base = np.stack([mri_slice]*3, axis=-1)  # gray to RGB
    overlay = base.copy()

    for cls, color in brats_colors.items():
        overlay[mask == cls] = np.array(color)

    return (1-alpha)*base + alpha*overlay

def compute_error_map(gt_mask, pred_mask):
    """
    Error map: 
      TP = white, FN = red, FP = blue
    """
    error_map = np.zeros((*gt_mask.shape, 3), dtype=np.float32)

    tp = (gt_mask > 0) & (pred_mask > 0)
    fn = (gt_mask > 0) & (pred_mask == 0)
    fp = (gt_mask == 0) & (pred_mask > 0)

    error_map[tp] = (1, 1, 1)   # white
    error_map[fn] = (1, 0, 0)   # red
    error_map[fp] = (0, 0, 1)   # blue

    return error_map

def visualize_overlays_with_errors(h5_file, test_idx, model, n_patients=3):
    """
    Show qualitative results: [FLAIR | GT overlay | Prediction overlay | Error map]
    for unique patients from test_idx.
    """
    brain_names = [name.decode("utf-8") for name in h5_file["brain_names"][:]]
    shown = 0

    for pid in test_idx:
        vol = h5_file["data"][pid]   # [X,Y,Z,C]
        gt  = h5_file["truth"][pid]  # [X,Y,Z]
        patient_name = brain_names[pid]

        # Pick slice with most tumor
        tumor_sums = np.sum(gt > 0, axis=(0,1))
        slice_idx = np.argmax(tumor_sums)

        flair = vol[:,:,slice_idx,0]
        gt_mask = gt[:,:,slice_idx]

        # Normalize & rotate
        flair_disp = (flair - flair.min())/(flair.max()-flair.min()+1e-8)
        flair_disp = np.rot90(flair_disp, k=3)
        gt_mask    = np.rot90(gt_mask,    k=3)

        # Predict
        inp = vol[:,:,slice_idx,:]
        inp = normalize_like_generator(inp)  # reuse your generator norm
        pred = model.predict(np.expand_dims(inp,0), verbose=0)
        pred_mask = np.argmax(pred[0], axis=-1)
        pred_mask = np.rot90(pred_mask, k=3)

        # Overlays
        gt_overlay   = overlay_mask_on_mri(flair_disp, gt_mask, alpha=0.4)
        pred_overlay = overlay_mask_on_mri(flair_disp, pred_mask, alpha=0.4)
        error_map    = compute_error_map(gt_mask, pred_mask)

        # Plot row
        fig, axs = plt.subplots(1,4, figsize=(10,4))
        axs[0].imshow(flair_disp, cmap="gray"); axs[0].set_title("FLAIR")
        axs[1].imshow(gt_overlay); axs[1].set_title("GT Overlay")
        axs[2].imshow(pred_overlay); axs[2].set_title("Prediction Overlay")
        axs[3].imshow(error_map); axs[3].set_title("Error Map")

        for a in axs: a.axis("off")

        # Row caption with Patient ID
        fig.text(0.01, 0.98, f"Patient: {patient_name}", 
                 ha="left", va="top", fontsize=9)

        # Legend (bottom-centered)
        legend_elements = [
            mpatches.Patch(color='yellow', label='NCR'),
            mpatches.Patch(color='green',  label='ED'),
            mpatches.Patch(color='red',    label='ET'),
            mpatches.Patch(color='white',  label='TP'),
            mpatches.Patch(color='blue',   label='FP'),
            mpatches.Patch(color='red',    label='FN'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', 
                   ncol=6, frameon=False)

        plt.tight_layout(pad=2.0)
        plt.show()

        shown += 1
        if shown >= n_patients:
            break





