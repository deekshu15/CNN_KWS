import pandas as pd
import numpy as np
from tqdm import tqdm


def compute_iou(gt_s, gt_e, pr_s, pr_e):
    inter = max(0.0, min(gt_e, pr_e) - max(gt_s, pr_s))
    union = max(gt_e, pr_e) - min(gt_s, pr_s)
    return inter / union if union > 0 else 0.0


def evaluate_kws_accuracy(
    inferencer,
    metadata_csv,
    max_samples=None
):
    df = pd.read_csv(metadata_csv)

    if len(df) == 0:
        raise ValueError("Metadata CSV is empty")

    if max_samples:
        df = df.sample(max_samples, random_state=42)

    missed = 0
    start_err = []
    end_err = []
    mae = []
    ious = []
    coverage = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        gt_start = float(row.start_time)
        gt_end = float(row.end_time)
        gt_dur = gt_end - gt_start

        result = inferencer.infer(row.audio_path, row.keyword)

        if result is None:
            missed += 1
            continue

        pr_start = result["start"]
        pr_end = result["end"]

        # Errors
        start_err.append(abs(pr_start - gt_start))
        end_err.append(abs(pr_end - gt_end))
        mae.append((abs(pr_start - gt_start) + abs(pr_end - gt_end)) / 2)

        # IoU
        iou = compute_iou(gt_start, gt_end, pr_start, pr_end)
        ious.append(iou)

        # Coverage
        inter = max(0.0, min(gt_end, pr_end) - max(gt_start, pr_start))
        coverage.append(inter / gt_dur if gt_dur > 0 else 0)

    total = len(df)

    return {
        "samples_evaluated": total,
        "miss_rate_%": round(missed / total * 100, 2),
        "mean_start_error_sec": round(np.mean(start_err), 3),
        "mean_end_error_sec": round(np.mean(end_err), 3),
        "mean_mae_sec": round(np.mean(mae), 3),
        "mean_iou": round(np.mean(ious), 3),
        "mean_coverage": round(np.mean(coverage), 3),
    }