"""
Data loading and preprocessing utilities for the HAM10000 dataset.
Uses ImageDataGenerator.flow_from_dataframe() for memory-efficient training.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config import (
    METADATA_CSV, IMAGE_DIR_PART1, IMAGE_DIR_PART2,
    IMAGE_SIZE, BATCH_SIZE, RANDOM_STATE, TEST_SIZE, VAL_TEST_SPLIT,
    ROTATION_RANGE, ZOOM_RANGE, WIDTH_SHIFT_RANGE, HEIGHT_SHIFT_RANGE,
    HORIZONTAL_FLIP, FILL_MODE,
)


def load_metadata(metadata_path=None):
    """Load HAM10000 metadata CSV and create image paths."""
    if metadata_path is None:
        metadata_path = METADATA_CSV

    df = pd.read_csv(metadata_path)
    print(f"Metadata loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"Missing values:\n{df.isnull().sum()}")

    # Create image path column
    df["image_path"] = df["image_id"].apply(_resolve_image_path)

    # Verify paths exist
    valid_mask = df["image_path"].apply(os.path.exists)
    n_missing = (~valid_mask).sum()
    if n_missing > 0:
        print(f"WARNING: {n_missing} images not found on disk — filtering them out.")
        df = df[valid_mask].reset_index(drop=True)

    # Create cell_type alias for compatibility
    df["cell_type"] = df["dx"]

    return df


def _resolve_image_path(image_id):
    """Resolve the full filesystem path for a given image ID."""
    path1 = os.path.join(IMAGE_DIR_PART1, image_id + ".jpg")
    if os.path.exists(path1):
        return path1
    path2 = os.path.join(IMAGE_DIR_PART2, image_id + ".jpg")
    return path2


def encode_labels(df):
    """Fit a LabelEncoder and add an encoded_label column."""
    encoder = LabelEncoder()
    df["encoded_label"] = encoder.fit_transform(df["cell_type"])
    print("Class mapping:")
    for idx, cls in enumerate(encoder.classes_):
        print(f"  {idx}: {cls}")
    return df, encoder


def split_data(df):
    """
    Split into train/validation/test partitions without sharing lesion IDs.

    HAM10000 contains multiple images of some lesions. When lesion_id is
    available, grouped stratified folds prevent images of one lesion from
    appearing in both training and evaluation. Datasets without lesion IDs use
    a conventional stratified 70/15/15 image split.
    """
    if "lesion_id" in df.columns and df["lesion_id"].notna().all():
        test_splitter = StratifiedGroupKFold(
            n_splits=7, shuffle=True, random_state=RANDOM_STATE
        )
        train_val_idx, test_idx = next(
            test_splitter.split(df, df["cell_type"], groups=df["lesion_id"])
        )
        train_val_df = df.iloc[train_val_idx]
        test_df = df.iloc[test_idx]

        val_splitter = StratifiedGroupKFold(
            n_splits=6, shuffle=True, random_state=RANDOM_STATE
        )
        train_idx, val_idx = next(
            val_splitter.split(
                train_val_df,
                train_val_df["cell_type"],
                groups=train_val_df["lesion_id"],
            )
        )
        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]
    else:
        train_df, temp_df = train_test_split(
            df,
            test_size=TEST_SIZE,
            stratify=df["cell_type"],
            random_state=RANDOM_STATE,
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=VAL_TEST_SPLIT,
            stratify=temp_df["cell_type"],
            random_state=RANDOM_STATE,
        )

    split_method = "lesion-grouped" if "lesion_id" in df.columns and df["lesion_id"].notna().all() else "image-level"
    print(
        f"Split ({split_method}) — Train: {len(train_df)} | "
        f"Val: {len(val_df)} | Test: {len(test_df)}"
    )
    return train_df, val_df, test_df


def compute_weights(train_df):
    """Compute balanced class weights from training set."""
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["encoded_label"]),
        y=train_df["encoded_label"],
    )
    class_weights_dict = dict(enumerate(weights))
    print(f"Class weights: {class_weights_dict}")
    return class_weights_dict


def create_generators(
    train_df, val_df, test_df, model_type="custom_cnn", image_size=IMAGE_SIZE
):
    """
    Build Keras ImageDataGenerators for train/val/test.

    Parameters
    ----------
    model_type : str
        Retained for compatibility with callers. All generators return [0, 1]
        inputs; transfer-learning models rescale internally to MobileNetV2's range.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=ROTATION_RANGE,
        zoom_range=ZOOM_RANGE,
        width_shift_range=WIDTH_SHIFT_RANGE,
        height_shift_range=HEIGHT_SHIFT_RANGE,
        horizontal_flip=HORIZONTAL_FLIP,
        fill_mode=FILL_MODE,
    )

    val_test_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
    )

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="image_path",
        y_col="cell_type",
        target_size=image_size,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )

    val_gen = val_test_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col="image_path",
        y_col="cell_type",
        target_size=image_size,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    test_gen = val_test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="image_path",
        y_col="cell_type",
        target_size=image_size,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    return train_gen, val_gen, test_gen
