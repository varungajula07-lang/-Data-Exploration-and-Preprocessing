"""
Cognifyz Technologies Data Science Internship - Level 1
Module: Data Loading & Dataset Overview
Author: Antigravity / Varun
"""

import os
import sys
import pandas as pd

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def detect_dataset_path(preferred_path: str = None) -> str:
    """
    Automatically detects the restaurant dataset CSV file path.
    Searches preferred path, standard project directories, and current directory.
    """
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    candidates = [
        os.path.join("data", "dataset.csv"),
        os.path.join("data", "Dataset.csv"),
        "dataset.csv",
        "Dataset.csv",
        os.path.join("..", "data", "dataset.csv"),
        os.path.join("..", "data", "Dataset.csv"),
        os.path.join("..", "dataset.csv"),
        os.path.join("..", "Dataset.csv"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)

    # Search current directory and subdirectories for any CSV file with 'dataset' or 'restaurant'
    search_dirs = [".", "data", ".."]
    for sdir in search_dirs:
        if os.path.exists(sdir):
            for root, _, files in os.walk(sdir):
                for f in files:
                    fl = f.lower()
                    if fl.endswith(".csv") and ("dataset" in fl or "restaurant" in fl or "zomato" in fl):
                        return os.path.abspath(os.path.join(root, f))

    raise FileNotFoundError(
        "Could not automatically locate the restaurant dataset CSV file. "
        "Please ensure 'dataset.csv' exists in the 'data/' directory."
    )


def load_dataset(filepath: str = None) -> pd.DataFrame:
    """
    Loads the restaurant dataset using robust encoding detection.
    Handles UTF-8 BOM, latin-1, and ISO-8859-1 encodings.
    """
    resolved_path = detect_dataset_path(filepath)
    print(f"[INFO] Loading dataset from: {resolved_path}")

    encodings_to_try = ["utf-8-sig", "latin-1", "iso-8859-1", "cp1252", "utf-8"]
    df = None

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(resolved_path, encoding=enc)
            print(f"[INFO] Successfully loaded dataset using '{enc}' encoding.")
            break
        except (UnicodeDecodeError, Exception) as e:
            continue

    if df is None:
        raise ValueError(f"Failed to load dataset from {resolved_path} with standard encodings.")

    # Strip any accidental leading/trailing whitespaces in column names
    df.columns = [c.strip() for c in df.columns]

    return df


def display_dataset_overview(df: pd.DataFrame) -> dict:
    """
    Displays and returns the primary structural overview of the dataset:
    - Number of rows
    - Number of columns
    - Column names
    - Data types
    - First 5 rows
    - Last 5 rows
    - Basic dataset information
    """
    num_rows, num_cols = df.shape

    print("=" * 70)
    print("DATASET STRUCTURAL OVERVIEW")
    print("=" * 70)
    print(f"Total Number of Rows    : {num_rows:,}")
    print(f"Total Number of Columns : {num_cols}")
    print("\nColumn Names & Data Types:")
    print("-" * 70)
    col_info = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.values,
        "Non-Null Count": df.notnull().sum().values
    })
    print(col_info.to_string(index=False))

    print("\n" + "=" * 70)
    print("FIRST 5 ROWS (HEAD):")
    print("=" * 70)
    print(df.head(5).to_string())

    print("\n" + "=" * 70)
    print("LAST 5 ROWS (TAIL):")
    print("=" * 70)
    print(df.tail(5).to_string())

    overview = {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "head": df.head(5),
        "tail": df.tail(5)
    }

    return overview


if __name__ == "__main__":
    df = load_dataset()
    overview = display_dataset_overview(df)
