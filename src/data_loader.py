import pandas as pd

FILE_PATH = r"data/data_set.csv"


def data_load(file_path:str =FILE_PATH)->pd.DataFrame:
    """
    Load dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    try:
        return pd.read_csv(file_path)
    
    except FileNotFoundError as e:

        print(f"The data_set not found ! {e}")
        raise

def main()-> None:
    print(data_load())


if __name__ == "__main__":
    main()
