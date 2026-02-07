import pandas as pd
import numpy as np
from tqdm import tqdm


def compute_iou(gt_s, gt_e, pr_s, pr_e):
    inter = max(0.0, min(gt_e, pr_e) - max(gt_s, pr_s))
    union = max(gt_e, pr_e) - min(gt_s, pr_s)
    return inter / union if union > 0 else 0.0


def coverage(gt_s, gt_e, pr_s, pr_e):
    inter = max(0.0, min(gt_e, pr_e) - max(gt_s, pr_s))
    return inter / (gt_e - gt_s) if gt_e > gt_s else 0.0


def evaluate_folder(
    inferencer,
    metadata_csv,
    max_samples=None
):
    df = pd.read_csv(metadata_csv)

    if len(df) == 0:
        raise ValueError("Metadata file is empty")

    start_err, end_err, mae = [], [], []
    ious, coverages = [], []
    missed = 0
    total = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        wav = row["audio_path"]
        kw = row["keyword"]
        gt_s = float(row["start_time"])
        gt_e = float(row["end_time"])

        pred = inferencer.infer(wav, kw)
        total += 1

        if pred is None:
            missed += 1
            continue

        ps, pe, conf = pred

        se = abs(ps - gt_s)
        ee = abs(pe - gt_e)

        start_err.append(se)
        end_err.append(ee)
        mae.append((se + ee) / 2)

        ious.append(compute_iou(gt_s, gt_e, ps, pe))
        coverages.append(coverage(gt_s, gt_e, ps, pe))

        if max_samples and total >= max_samples:
            break

    return {
        "samples": total,
        "miss_rate_%": round(missed / total * 100, 2),
        "mean_start_error_sec": round(np.mean(start_err), 3),
        "mean_end_error_sec": round(np.mean(end_err), 3),
        "mean_mae_sec": round(np.mean(mae), 3),
        "mean_iou": round(np.mean(ious), 3),
        "mean_coverage": round(np.mean(coverages), 3),
    }
