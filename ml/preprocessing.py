"""
Feature engineering and preprocessing functions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from imblearn.over_sampling import SMOTE
import category_encoders as ce
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def filter_rare_classes(df: pd.DataFrame,
                        target_column: str,
                        min_samples: int = 50) -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Filter out rare classes from target variable
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataframe
    target_column : str
        Name of target column
    min_samples : int
        Minimum samples required for a class
        
    Returns:
    --------
    tuple
        (filtered_df, filtered_target, valid_classes)
    """
    class_counts = df[target_column].value_counts()
    logger.info(f"Original class distribution:\n{class_counts}")
    
    valid_classes = class_counts[class_counts >= min_samples].index.tolist()
    
    # Filter data
    mask = df[target_column].isin(valid_classes)
    df_filtered = df[mask].copy()
    target_filtered = df_filtered[target_column].copy()
    
    logger.info(f"Classes retained: {valid_classes}")
    logger.info(f"Samples after filtering: {len(df_filtered)}")
    
    return df_filtered, target_filtered, valid_classes


def prepare_features_and_targets(df: pd.DataFrame,
                                 temp_column: str,
                                 precip_column: str,
                                 min_samples: int = 50) -> Dict:
    """
    Prepare features and targets for classification and regression
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataframe
    temp_column : str
        Temperature column name
    precip_column : str
        Precipitation column name
    min_samples : int
        Minimum samples for classification classes
        
    Returns:
    --------
    dict
        Dictionary containing X, y_reg, y_clf, valid_classes
    """
    # Filter rare classes
    df_filtered, y_clf, valid_classes = filter_rare_classes(
        df, precip_column, min_samples
    )
    
    # Extract targets
    y_reg = df_filtered[temp_column].copy()
    
    # Prepare features (exclude target columns)
    X = df_filtered.drop(columns=[temp_column, precip_column])
    
    return {
        'X': X,
        'y_reg': y_reg,
        'y_clf': y_clf,
        'valid_classes': valid_classes
    }


def create_train_test_splits(X: pd.DataFrame,
                             y_reg: pd.Series,
                             y_clf: pd.Series,
                             test_size: float = 0.2,
                             random_state: int = 42) -> Dict:
    """
    Create train-test splits for both classification and regression
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y_reg : pd.Series
        Regression target
    y_clf : pd.Series
        Classification target
    test_size : float
        Test set proportion
    random_state : int
        Random seed
        
    Returns:
    --------
    dict
        Dictionary with train-test splits
    """
    # Classification split (stratified)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X, y_clf, test_size=test_size, random_state=random_state, stratify=y_clf
    )
    
    # Regression split
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Train/test split: {X_train_clf.shape[0]} train, {X_test_clf.shape[0]} test")
    
    return {
        'classification': {
            'X_train': X_train_clf,
            'X_test': X_test_clf,
            'y_train': y_train_clf,
            'y_test': y_test_clf
        },
        'regression': {
            'X_train': X_train_reg,
            'X_test': X_test_reg,
            'y_train': y_train_reg,
            'y_test': y_test_reg
        }
    }


def create_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Create preprocessing pipeline
    - StandardScaler for numeric features
    - CountEncoder for categorical features
    
    Parameters:
    -----------
    numeric_cols : list
        Numeric column names
    categorical_cols : list
        Categorical column names
        
    Returns:
    --------
    ColumnTransformer
        Fitted preprocessor
    """
    transformers = []
    
    if len(numeric_cols) > 0:
        transformers.append(('num', StandardScaler(), numeric_cols))
    
    if len(categorical_cols) > 0:
        transformers.append(('cat', ce.CountEncoder(), categorical_cols))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    
    logger.info(f"Preprocessor created with {len(transformers)} transformers")
    
    return preprocessor


def preprocess_train_test_data(splits: Dict,
                               numeric_cols: list,
                               categorical_cols: list,
                               apply_smote: bool = True) -> Dict:
    """
    Apply preprocessing to train and test data
    
    Parameters:
    -----------
    splits : dict
        Train-test splits
    numeric_cols : list
        Numeric column names
    categorical_cols : list
        Categorical column names
    apply_smote : bool
        Whether to apply SMOTE to classification data
        
    Returns:
    --------
    dict
        Preprocessed data and fitted preprocessor
    """
    # Update column lists (remove target columns if present)
    target_cols = ['Temperature (C)', 'Precip Type']
    numeric_cols = [col for col in numeric_cols if col not in target_cols]
    categorical_cols = [col for col in categorical_cols if col not in target_cols]
    
    # Create preprocessor
    preprocessor = create_preprocessor(numeric_cols, categorical_cols)
    
    # Classification preprocessing
    X_train_clf_proc = preprocessor.fit_transform(splits['classification']['X_train'])
    X_test_clf_proc = preprocessor.transform(splits['classification']['X_test'])
    y_train_clf = splits['classification']['y_train'].copy()
    y_test_clf = splits['classification']['y_test']
    
    # Apply SMOTE
    if apply_smote:
        smote = SMOTE(random_state=42)
        X_train_clf_proc, y_train_clf = smote.fit_resample(X_train_clf_proc, y_train_clf)
        logger.info("SMOTE applied to classification training data")
    
    # Regression preprocessing
    X_train_reg_proc = preprocessor.fit_transform(splits['regression']['X_train'])
    X_test_reg_proc = preprocessor.transform(splits['regression']['X_test'])
    
    logger.info("Preprocessing complete")
    
    return {
        'classification': {
            'X_train': X_train_clf_proc,
            'X_test': X_test_clf_proc,
            'y_train': y_train_clf,
            'y_test': y_test_clf
        },
        'regression': {
            'X_train': X_train_reg_proc,
            'X_test': X_test_reg_proc,
            'y_train': splits['regression']['y_train'],
            'y_test': splits['regression']['y_test']
        },
        'preprocessor': preprocessor,
        'feature_names': numeric_cols + categorical_cols,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols
    }


def perform_clustering(df: pd.DataFrame,
                      cluster_features: list,
                      n_clusters: int = 4,
                      random_state: int = 42) -> Tuple[np.ndarray, Dict]:
    """
    Perform K-Means clustering on specified features
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe
    cluster_features : list
        Features to use for clustering
    n_clusters : int
        Number of clusters
    random_state : int
        Random seed
        
    Returns:
    --------
    tuple
        (cluster_labels, clustering_metrics)
    """
    # Prepare data
    cluster_data = df[cluster_features].fillna(df[cluster_features].median())
    
    # Scale features
    scaler = StandardScaler()
    cluster_data_scaled = scaler.fit_transform(cluster_data)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(cluster_data_scaled)
    
    # Calculate metrics
    silhouette = silhouette_score(cluster_data_scaled, cluster_labels)
    davies_bouldin = davies_bouldin_score(cluster_data_scaled, cluster_labels)
    
    metrics = {
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'n_clusters': n_clusters
    }
    
    logger.info(f"K-Means Clustering Results:")
    logger.info(f"  Silhouette Score: {silhouette:.3f}")
    logger.info(f"  Davies-Bouldin Score: {davies_bouldin:.3f}")
    
    return cluster_labels, metrics