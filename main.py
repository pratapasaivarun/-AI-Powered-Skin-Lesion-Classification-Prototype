"""
main.py — CLI entry point for the Skin Cancer Detection System.

Usage:
    python main.py --mode train    --model hybrid
    python main.py --mode evaluate --model hybrid
    python main.py --mode predict  --image path/to/image.jpg
    python main.py --mode download                           # download HAM10000
"""

import argparse
import os
import sys
import pickle

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.console import configure_console

configure_console()

from src.config import (
    MODEL_TYPES, DATASET_DIR, METADATA_CSV,
    FINAL_MODEL_PATH, LABEL_ENCODER_PATH, TRAINING_HISTORY_PATH,
    IMAGE_DIR_PART1, IMAGE_DIR_PART2,
)


def download_dataset():
    """Download HAM10000 from Kaggle (requires kaggle.json configured)."""
    import subprocess

    print("=" * 60)
    print("  Downloading HAM10000 dataset from Kaggle...")
    print("=" * 60)
    os.makedirs(DATASET_DIR, exist_ok=True)

    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d",
             "kmader/skin-cancer-mnist-ham10000", "-p", DATASET_DIR],
            check=True,
        )
        # Unzip
        zip_path = os.path.join(DATASET_DIR, "skin-cancer-mnist-ham10000.zip")
        if os.path.exists(zip_path):
            import zipfile
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(DATASET_DIR)
            os.remove(zip_path)
            print(f"Dataset extracted to {DATASET_DIR}")
        else:
            print("Download completed but zip file not found. Check manually.")
    except FileNotFoundError:
        print("ERROR: 'kaggle' CLI not found. Install it with:")
        print("  pip install kaggle")
        print("Then place your kaggle.json in ~/.kaggle/")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Kaggle download failed: {e}")
        sys.exit(1)


def run_train(model_type):
    """Full training pipeline."""
    from src.data_loader import load_metadata, encode_labels, split_data, compute_weights, create_generators
    from src.train import train_model
    from src.evaluate import plot_training_curves, evaluate_model

    print(f"\n{'=' * 60}")
    print(f"  TRAINING: {model_type.upper()}")
    print(f"{'=' * 60}\n")

    # 1. Load data
    df = load_metadata()
    df, encoder = encode_labels(df)
    train_df, val_df, test_df = split_data(df)
    class_weights = compute_weights(train_df)

    # 2. Create generators
    gen_type = "mobilenet_v2" if model_type == "mobilenet_v2" else "custom_cnn"
    train_gen, val_gen, test_gen = create_generators(train_df, val_df, test_df, model_type=gen_type)

    # 3. Train
    model, history = train_model(model_type, train_gen, val_gen, class_weights, encoder)

    # 4. Plot curves
    plot_training_curves(history)

    # 5. Evaluate on test set
    metrics = evaluate_model(model, test_gen, list(encoder.classes_))

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"  F1 Score:      {metrics['f1']:.4f}")
    print("=" * 60)

    return model, metrics


def run_evaluate():
    """Load a saved model and evaluate on the test set."""
    from src.data_loader import load_metadata, encode_labels, split_data, create_generators
    from src.predict import load_artifacts
    from src.evaluate import evaluate_model

    model, encoder = load_artifacts()
    df = load_metadata()
    df, _ = encode_labels(df)
    _, _, test_df = split_data(df)

    _, _, test_gen = create_generators(test_df, test_df, test_df)
    metrics = evaluate_model(model, test_gen, list(encoder.classes_))
    return metrics


def run_predict(image_path):
    """Predict on a single image."""
    from src.predict import load_artifacts, predict_single_image

    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        sys.exit(1)

    model, encoder = load_artifacts()
    result = predict_single_image(image_path, model, encoder)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Skin Cancer Detection & Triage System",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "evaluate", "predict", "download"],
        help=(
            "Mode of operation:\n"
            "  train    — Train a model from scratch\n"
            "  evaluate — Evaluate a saved model on the test set\n"
            "  predict  — Predict a single image\n"
            "  download — Download HAM10000 dataset from Kaggle"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="hybrid",
        choices=MODEL_TYPES,
        help=f"Model architecture (default: hybrid). Options: {MODEL_TYPES}",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to skin lesion image (required for --mode predict)",
    )

    args = parser.parse_args()

    if args.mode == "download":
        download_dataset()

    elif args.mode == "train":
        # Verify dataset exists
        if not os.path.exists(METADATA_CSV):
            print(f"ERROR: Dataset not found at {METADATA_CSV}")
            print("Run:  python main.py --mode download")
            sys.exit(1)
        run_train(args.model)

    elif args.mode == "evaluate":
        if not os.path.exists(FINAL_MODEL_PATH):
            print(f"ERROR: No saved model at {FINAL_MODEL_PATH}")
            print("Train first:  python main.py --mode train")
            sys.exit(1)
        run_evaluate()

    elif args.mode == "predict":
        if args.image is None:
            print("ERROR: --image is required for predict mode")
            sys.exit(1)
        if not os.path.exists(FINAL_MODEL_PATH):
            print(f"ERROR: No saved model at {FINAL_MODEL_PATH}")
            print("Train first:  python main.py --mode train")
            sys.exit(1)
        run_predict(args.image)


if __name__ == "__main__":
    main()
