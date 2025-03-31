from zenml import step, pipeline
import pandas as pd
import csv
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

# ----- Step 1: Clean Raw Data -----
@step
def clean_data( file_path = '../data/multimodal/raw/COMP5329S1A2Dataset/train.csv') -> pd.DataFrame:
    """Clean raw CSV data and handle malformed lines"""
    correct_lines = []
    problematic_lines = []

    # Read and process raw file
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, quotechar='"', delimiter=',', 
                          doublequote=True, skipinitialspace=True)
        
        # Process header
        header = next(reader)
        correct_lines.append(header)
        
        # Process rows
        for line_number, fields in enumerate(reader, start=2):
            if len(fields) == 3:
                correct_lines.append(fields)
            else:
                print(f"Problematic line {line_number}: {fields}")
                problematic_lines.append(fields)

    # Fix problematic lines
    for fields in problematic_lines:
        if len(fields) > 3:
            # Merge extra columns into Caption
            fields = [fields[0], fields[1], ','.join(fields[2:])]
        elif len(fields) < 3:
            # Pad missing columns
            fields += [''] * (3 - len(fields))
        correct_lines.append(fields)

    # Create DataFrame with proper types
    df = pd.DataFrame(correct_lines[1:], columns=header)
    
    return df

# ----- Step 2: Split Data -----
@step
def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the DataFrame into train, validation, and test subsets."""
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))
    return (
        df[:train_size],
        df[train_size : train_size + val_size],
        df[train_size + val_size :],
    )

@step
def encode_labels(
    train: pd.DataFrame, 
    val: pd.DataFrame, 
    test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MultiLabelBinarizer]:
    """Encode labels for train, validation, and test sets."""
    mlb = MultiLabelBinarizer()
    
    train_labels = mlb.fit_transform(
        train["Labels"].str.split().apply(lambda x: list(map(int, x)))
    )
    val_labels = mlb.transform(
        val["Labels"].str.split().apply(lambda x: list(map(int, x)))
    )
    test_labels = mlb.transform(
        test["Labels"].str.split().apply(lambda x: list(map(int, x)))
    )
    
    return train_labels, val_labels, test_labels, mlb


# ----- Main Pipeline -----
@pipeline
def data_processing_pipeline(clean_step, split_step, encode_step):
    """Define the pipeline workflow."""
    # Clean the data
    df = clean_step()
    
    # Split the data into train, validation, and test
    train, val, test = split_step(df)
    
    # Encode the labels
    encode_step(train, val, test)


