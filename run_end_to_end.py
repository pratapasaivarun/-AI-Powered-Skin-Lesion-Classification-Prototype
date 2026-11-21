"""
run_end_to_end.py — Full end-to-end pipeline: unzip → train → evaluate → predict → report
"""

import os
import sys
import time
import json
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    DATASET_DIR, METADATA_CSV, IMAGE_DIR_PART1, IMAGE_DIR_PART2,
    FINAL_MODEL_PATH, LABEL_ENCODER_PATH, RESULTS_DIR,
)
from src.console import configure_console

configure_console()


def step_banner(title):
    print(f"\n{'=' * 70}")
    print(f"  STEP: {title}")
    print(f"{'=' * 70}\n")


def main():
    overall_start = time.time()
    results_log = {}

    # ============================================================
    # STEP 0: Unzip dataset if needed
    # ============================================================
    step_banner("EXTRACT DATASET")
    zip_path = os.path.join(DATASET_DIR, "skin-cancer-mnist-ham10000.zip")
    if os.path.exists(zip_path) and not os.path.exists(METADATA_CSV):
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATASET_DIR)
        print("Extraction complete.")
    elif os.path.exists(METADATA_CSV):
        print("Dataset already extracted.")
    else:
        print(f"ERROR: No zip file at {zip_path} and no metadata CSV found.")
        sys.exit(1)

    # Verify
    if not os.path.exists(METADATA_CSV):
        print(f"ERROR: {METADATA_CSV} not found after extraction.")
        sys.exit(1)

    img_count_p1 = len([f for f in os.listdir(IMAGE_DIR_PART1) if f.endswith('.jpg')]) if os.path.isdir(IMAGE_DIR_PART1) else 0
    img_count_p2 = len([f for f in os.listdir(IMAGE_DIR_PART2) if f.endswith('.jpg')]) if os.path.isdir(IMAGE_DIR_PART2) else 0
    total_images = img_count_p1 + img_count_p2
    print(f"Images found: Part1={img_count_p1}, Part2={img_count_p2}, Total={total_images}")
    results_log["dataset"] = {"part1": img_count_p1, "part2": img_count_p2, "total": total_images}

    # ============================================================
    # STEP 1: Load & prepare data
    # ============================================================
    step_banner("LOAD & PREPARE DATA")
    from src.data_loader import load_metadata, encode_labels, split_data, compute_weights, create_generators

    df = load_metadata()
    results_log["metadata_shape"] = list(df.shape)
    print(f"\nClass distribution:\n{df['dx'].value_counts().to_string()}")
    results_log["class_distribution"] = df['dx'].value_counts().to_dict()

    df, encoder = encode_labels(df)
    train_df, val_df, test_df = split_data(df)
    results_log["split"] = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}

    class_weights = compute_weights(train_df)
    results_log["class_weights"] = {str(k): round(v, 4) for k, v in class_weights.items()}

    # Use custom_cnn for local CPU (faster than hybrid/mobilenet for demo)
    model_type = "custom_cnn"
    gen_type = "custom_cnn"
    train_gen, val_gen, test_gen = create_generators(train_df, val_df, test_df, model_type=gen_type)

    # ============================================================
    # STEP 2: Build & Train (reduced epochs for local CPU)
    # ============================================================
    step_banner(f"TRAIN MODEL ({model_type.upper()}) — CPU Mode")

    # Override epochs for CPU feasibility
    import src.config as cfg
    cfg.EPOCHS_STAGE1 = 5    # Reduced from 40 for CPU
    cfg.EPOCHS_STAGE2 = 0    # Skip fine-tuning (custom_cnn doesn't use it anyway)

    from src.train import train_model
    train_start = time.time()
    model, history = train_model(model_type, train_gen, val_gen, class_weights, encoder)
    train_time = time.time() - train_start
    print(f"\nTraining time: {train_time:.1f}s ({train_time/60:.1f} min)")

    results_log["training"] = {
        "model_type": model_type,
        "epochs_run": len(history["accuracy"]),
        "final_train_acc": round(history["accuracy"][-1], 4),
        "final_val_acc": round(history["val_accuracy"][-1], 4),
        "final_train_loss": round(history["loss"][-1], 4),
        "final_val_loss": round(history["val_loss"][-1], 4),
        "best_val_acc": round(max(history["val_accuracy"]), 4),
        "training_time_seconds": round(train_time, 1),
    }

    # ============================================================
    # STEP 3: Plot training curves
    # ============================================================
    step_banner("PLOT TRAINING CURVES")
    from src.evaluate import plot_training_curves, evaluate_model
    curves_path = plot_training_curves(history)
    results_log["training_curves_path"] = curves_path

    # ============================================================
    # STEP 4: Evaluate on test set
    # ============================================================
    step_banner("EVALUATE ON TEST SET")
    eval_start = time.time()
    metrics = evaluate_model(model, test_gen, list(encoder.classes_))
    eval_time = time.time() - eval_start

    results_log["evaluation"] = {
        "test_accuracy": round(metrics["test_accuracy"], 4),
        "test_loss": round(metrics["test_loss"], 4),
        "weighted_precision": round(metrics["precision"], 4),
        "weighted_recall": round(metrics["recall"], 4),
        "weighted_f1": round(metrics["f1"], 4),
        "classification_report": metrics["classification_report"],
        "evaluation_time_seconds": round(eval_time, 1),
    }

    # ============================================================
    # STEP 5: Single image prediction + Grad-CAM
    # ============================================================
    step_banner("SINGLE IMAGE PREDICTION + GRAD-CAM")
    from src.predict import predict_single_image

    # Pick a sample test image
    sample_path = test_df["image_path"].iloc[0]
    print(f"Predicting: {sample_path}")
    pred_result = predict_single_image(sample_path, model, encoder)

    results_log["sample_prediction"] = {
        "image": os.path.basename(pred_result["image_path"]),
        "predicted_class": pred_result["predicted_class"],
        "predicted_name": pred_result["predicted_name"],
        "confidence": round(pred_result["confidence"], 4),
        "risk_level": pred_result["risk_level"],
        "probabilities": {k: round(v, 4) for k, v in pred_result["probabilities"].items()},
    }

    # ============================================================
    # STEP 6: Summary
    # ============================================================
    total_time = time.time() - overall_start
    results_log["total_time_seconds"] = round(total_time, 1)
    results_log["total_time_minutes"] = round(total_time / 60, 1)

    step_banner("END-TO-END COMPLETE")
    print(f"Total pipeline time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Test Accuracy:       {results_log['evaluation']['test_accuracy']:.4f}")
    print(f"Weighted F1:         {results_log['evaluation']['weighted_f1']:.4f}")
    print(f"Weighted Precision:  {results_log['evaluation']['weighted_precision']:.4f}")
    print(f"Weighted Recall:     {results_log['evaluation']['weighted_recall']:.4f}")

    # Save results JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "run_results.json")
    with open(results_path, "w") as f:
        json.dump(results_log, f, indent=2, default=str)
    print(f"\nFull results saved → {results_path}")

    return results_log


if __name__ == "__main__":
    main()
