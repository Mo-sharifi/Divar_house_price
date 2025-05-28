from data_loader import data_load
import numpy as np
import pandas as pd
import time

pd.options.display.float_format = "{:,.0f}".format
from pathlib import Path


def convert_format(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Format specified columns: remove commas & convert 'Area' to int, bool columns to int8.

    Args:
        df (pd.DataFrame): Input data.
        columns (list): Columns to convert.

    Returns:
        pd.DataFrame: Formatted DataFrame.

    Raises:
        KeyError: Missing columns.
        ValueError: Conversion errors.
    """
    if columns is None:
        columns = ["Area", "Parking", "Warehouse", "Elevator"]

    try:
        for i in columns:
            if i == "Area":
                df["Area"] = df["Area"].str.replace(",", "").astype(int)
            else:
                df[i] = df[i].astype("int8")
        return df

    except Exception as e:

        print(f"ERROR in convert_format {e}")

        raise


def remove_missing_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove all rows with missing values from the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame without any missing values.

    Raises:
        Exception: If dropping missing values fails.
    """

    try:
        return df.dropna()

    except Exception as e:

        print("ERROR in remove_missing_value {e}")

        raise


def remove_outliers(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Remove outliers from specified columns using the IQR method.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list[str] | None): Columns to apply outlier removal.

    Returns:
        pd.DataFrame: DataFrame without outliers.

    Raises:
        KeyError, ValueError: If column not found or invalid values.
    """

    if columns is None:
        columns = ["Area", "Price"]
    try:

        for col in columns:

            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]

        return df

    except Exception as e:
        print("ERROR in remove_outliers {e}")
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    - Drop unnecessary columns
    - Format key columns
    - Remove missing values
    - Remove outliers

    Args:
        df (pd.DataFrame): Raw input data.

    Returns:
        pd.DataFrame: Cleaned data ready for modeling.

    Raises:
        Exception: If any step in the pipeline fails.
    """
    try:
        df_useful = df.drop(columns=["Price(USD)"], errors="ignore")
        df_converted = convert_format(df_useful)
        df_no_missing = remove_missing_value(df_converted)
        df_clean = remove_outliers(df_no_missing)

        return df_clean

    except Exception as e:

        print(f"ERROR in preprocessing_data {e}")

        raise


def save_clean_data(
    df: pd.DataFrame, path: str = "Divar_house_prediction/data/processed/"
):
    """
    Save cleaned DataFrame to CSV if not already saved.

    Args:
        df (pd.DataFrame): Cleaned data.
        path (str): Directory to save the file in.

    Raises:
        Exception: If file cannot be saved.
    """

    file_path = Path(path) / "clean_data.csv"
    try:
        if file_path.exists():

            print("Data was already saved before.")
        else:
            df.to_csv(file_path, index=False)

            print("Data set was cleaned and is now usable for training!")

    except Exception as e:
        print(f"ERROR in save_clean_data {e}")
        raise


def main() -> None:
    """
    Main execution pipeline for loading, preprocessing, and saving clean housing data.
    """
    
    try:
        df = data_load()
        cleaned_data = preprocess_data(df)
        save_clean_data(cleaned_data)
        print("Pipeline completed successfully")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
