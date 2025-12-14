"""
Data loading and initial cleaning functions
"""

import pandas as pd
import numpy as np
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw weather data from CSV file
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def initial_cleaning(df: pd.DataFrame, missing_threshold: float = 0.6) -> pd.DataFrame:
    """
    Perform initial data cleaning
    - Replace '?' with NaN
    - Drop rows with too many missing values
    - Remove duplicates
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataframe
    missing_threshold : float
        Threshold for dropping rows (default: 0.6)
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    df = df.copy()
    
    # Replace '?' with NaN
    df = df.replace('?', np.nan)
    
    # Drop rows with too many missing values
    initial_rows = len(df)
    df.dropna(thresh=int(missing_threshold * df.shape[1]), inplace=True)
    rows_dropped = initial_rows - len(df)
    logger.info(f"Dropped {rows_dropped} rows with excessive missing values")
    
    # Remove duplicates
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    duplicates_removed = initial_rows - len(df)
    logger.info(f"Removed {duplicates_removed} duplicate rows")
    
    return df


def extract_datetime_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Extract datetime features from date column
    - hour, day_of_week, month, season
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with date column
    date_column : str
        Name of the date column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with extracted datetime features
    """
    df = df.copy()
    
    # Convert to datetime
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce', utc=True)
    df = df.dropna(subset=[date_column])
    
    # Extract features
    df['hour'] = df[date_column].dt.hour
    df['day_of_week'] = df[date_column].dt.dayofweek
    df['month'] = df[date_column].dt.month
    df['season'] = df['month'].apply(lambda x: (x % 12 + 3) // 3)
    
    logger.info("Datetime features extracted: hour, day_of_week, month, season")
    
    return df


def drop_unnecessary_columns(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Drop columns not needed for modeling
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe
    date_column : str
        Name of date column to drop
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with unnecessary columns removed
    """
    df = df.copy()
    
    cols_to_drop = [date_column]
    if 'Daily Summary' in df.columns:
        cols_to_drop.append('Daily Summary')
    
    existing_cols = [col for col in cols_to_drop if col in df.columns]
    df.drop(columns=existing_cols, inplace=True)
    
    logger.info(f"Dropped columns: {existing_cols}")
    
    return df


def identify_column_types(df: pd.DataFrame) -> Tuple[list, list]:
    """
    Identify categorical and numeric columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe
        
    Returns:
    --------
    tuple
        (categorical_cols, numeric_cols)
    """
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
    numeric_cols = [col for col in df.columns if col not in categorical_cols]
    
    logger.info(f"Identified {len(categorical_cols)} categorical and {len(numeric_cols)} numeric columns")
    
    return categorical_cols, numeric_cols


def fill_missing_values(df: pd.DataFrame, 
                        categorical_cols: list, 
                        numeric_cols: list,
                        precip_column: str = 'Precip Type') -> pd.DataFrame:
    """
    Fill missing values
    - Numeric: median
    - Categorical: 'Missing', except Precip Type -> 'sunny'
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with missing values
    categorical_cols : list
        List of categorical column names
    numeric_cols : list
        List of numeric column names
    precip_column : str
        Name of precipitation column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with filled missing values
    """
    df = df.copy()
    
    # Fill numeric columns with median
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        logger.info(f"Filled missing values in {len(numeric_cols)} numeric columns with median")
    
    # Fill categorical columns
    if len(categorical_cols) > 0:
        df[categorical_cols] = df[categorical_cols].fillna('Missing')
        
        # Special handling for Precip Type
        if precip_column in df.columns:
            df[precip_column] = df[precip_column].replace('Missing', 'sunny')
            logger.info(f"Replaced 'Missing' with 'sunny' in {precip_column}")
    
    return df


def load_and_preprocess_data(filepath: str,
                             date_column: str,
                             missing_threshold: float = 0.6) -> Tuple[pd.DataFrame, list, list]:
    """
    Complete data loading and preprocessing pipeline
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
    date_column : str
        Name of date column
    missing_threshold : float
        Threshold for dropping rows
        
    Returns:
    --------
    tuple
        (preprocessed_df, categorical_cols, numeric_cols)
    """
    # Load data
    df = load_raw_data(filepath)
    
    # Initial cleaning
    df = initial_cleaning(df, missing_threshold)
    
    # Extract datetime features
    df = extract_datetime_features(df, date_column)
    
    # Drop unnecessary columns
    df = drop_unnecessary_columns(df, date_column)
    
    # Identify column types
    categorical_cols, numeric_cols = identify_column_types(df)
    
    # Fill missing values
    df = fill_missing_values(df, categorical_cols, numeric_cols)
    
    logger.info("Data loading and preprocessing complete")
    
    return df, categorical_cols, numeric_cols