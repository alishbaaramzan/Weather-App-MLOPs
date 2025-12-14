"""
Visualization functions for EDA and model results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import logging
import os

logger = logging.getLogger(__name__)


def save_plot(filename: str, output_dir: str = "plots", dpi: int = 300):
    """Helper function to save plots"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    logger.info(f"Plot saved: {filepath}")


def plot_eda_overview(df: pd.DataFrame,
                     temp_col: str,
                     precip_col: str,
                     output_dir: str = "plots"):
    """
    Create comprehensive EDA plots
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe
    temp_col : str
        Temperature column name
    precip_col : str
        Precipitation column name
    output_dir : str
        Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Temperature Distribution
    axes[0, 0].hist(df[temp_col], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].set_title('Temperature Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Temperature (°C)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Temperature by Season
    df.boxplot(column=temp_col, by='season', ax=axes[0, 1])
    axes[0, 1].set_title('Temperature by Season', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Season')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].set_xticklabels(['Winter', 'Spring', 'Summer', 'Fall'])
    plt.sca(axes[0, 1])
    
    # 3. Precipitation Type Distribution
    if precip_col in df.columns:
        precip_counts = df[precip_col].value_counts()
        axes[1, 0].bar(range(len(precip_counts)), precip_counts.values, 
                       edgecolor='black', color='coral')
        axes[1, 0].set_title('Precipitation Type Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Precipitation Type')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_xticks(range(len(precip_counts)))
        axes[1, 0].set_xticklabels(precip_counts.index, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Temperature by Hour
    hourly_temp = df.groupby('hour')[temp_col].mean()
    axes[1, 1].plot(hourly_temp.index, hourly_temp.values, 
                    marker='o', linewidth=2, color='darkgreen')
    axes[1, 1].set_title('Average Temperature by Hour', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Hour of Day')
    axes[1, 1].set_ylabel('Average Temperature (°C)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_plot('weather_eda_overview.png', output_dir)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame,
                             numeric_cols: list,
                             output_dir: str = "plots"):
    """
    Plot correlation heatmap
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe
    numeric_cols : list
        Numeric column names
    output_dir : str
        Directory to save plots
    """
    numeric_df = df[numeric_cols].select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    save_plot('weather_correlation.png', output_dir)
    plt.close()


def plot_clusters(df: pd.DataFrame,
                 cluster_col: str,
                 temp_col: str,
                 output_dir: str = "plots"):
    """
    Plot cluster analysis results
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with cluster labels
    cluster_col : str
        Cluster column name
    temp_col : str
        Temperature column name
    output_dir : str
        Directory to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Cluster distribution
    cluster_counts = df[cluster_col].value_counts().sort_index()
    axes[0].bar(cluster_counts.index, cluster_counts.values, 
                edgecolor='black', color='skyblue')
    axes[0].set_title('Cluster Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Cluster')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Temperature by cluster
    df.boxplot(column=temp_col, by=cluster_col, ax=axes[1])
    axes[1].set_title('Temperature Distribution by Cluster', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Temperature (°C)')
    plt.sca(axes[1])
    
    plt.tight_layout()
    save_plot('weather_clusters.png', output_dir)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray,
                         class_names: list,
                         output_dir: str = "plots"):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    cm : np.ndarray
        Confusion matrix
    class_names : list
        Class names
    output_dir : str
        Directory to save plots
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={"shrink": 0.8})
    plt.title('Confusion Matrix - Best Classifier', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    save_plot('confusion_matrix.png', output_dir)
    plt.close()


def plot_regression_predictions(y_true: np.ndarray,
                                y_pred: np.ndarray,
                                output_dir: str = "plots"):
    """
    Plot actual vs predicted values for regression
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    output_dir : str
        Directory to save plots
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10, color='steelblue')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Temperature (°C)', fontsize=12)
    plt.ylabel('Predicted Temperature (°C)', fontsize=12)
    plt.title('Actual vs Predicted Temperature', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot('regression_predictions.png', output_dir)
    plt.close()


def plot_feature_importance(importance_df_clf: pd.DataFrame,
                           importance_df_reg: pd.DataFrame,
                           output_dir: str = "plots"):
    """
    Plot feature importance for classifier and regressor
    
    Parameters:
    -----------
    importance_df_clf : pd.DataFrame
        Classifier feature importance
    importance_df_reg : pd.DataFrame
        Regressor feature importance
    output_dir : str
        Directory to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Classifier
    if importance_df_clf is not None:
        axes[0].barh(importance_df_clf['feature'], importance_df_clf['importance'], 
                     color='steelblue', edgecolor='black')
        axes[0].set_xlabel('Importance', fontsize=12)
        axes[0].set_title('Top Features - Classifier', fontsize=14, fontweight='bold')
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3, axis='x')
    
    # Regressor
    if importance_df_reg is not None:
        axes[1].barh(importance_df_reg['feature'], importance_df_reg['importance'],
                     color='coral', edgecolor='black')
        axes[1].set_xlabel('Importance', fontsize=12)
        axes[1].set_title('Top Features - Regressor', fontsize=14, fontweight='bold')
        axes[1].invert_yaxis()
        axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    save_plot('feature_importance.png', output_dir)
    plt.close()


def plot_model_comparison(clf_results: pd.DataFrame,
                         reg_results: pd.DataFrame,
                         output_dir: str = "plots"):
    """
    Plot comparison of initial models
    
    Parameters:
    -----------
    clf_results : pd.DataFrame
        Classification results
    reg_results : pd.DataFrame
        Regression results
    output_dir : str
        Directory to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Classification comparison
    clf_results['f1_macro'].plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
    axes[0].set_title('Classifier F1-Score Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Model', fontsize=12)
    axes[0].set_ylabel('F1 Score (Macro)', fontsize=12)
    axes[0].set_xticklabels(clf_results.index, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Regression comparison
    reg_results['r2'].plot(kind='bar', ax=axes[1], color='coral', edgecolor='black')
    axes[1].set_title('Regressor R² Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Model', fontsize=12)
    axes[1].set_ylabel('R² Score', fontsize=12)
    axes[1].set_xticklabels(reg_results.index, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_plot('model_comparison.png', output_dir)
    plt.close()


def create_all_visualizations(df: pd.DataFrame,
                              temp_col: str,
                              precip_col: str,
                              numeric_cols: list,
                              evaluation_results: Dict,
                              valid_classes: list,
                              output_dir: str = "plots"):
    """
    Create all visualizations in one function
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe
    temp_col : str
        Temperature column
    precip_col : str
        Precipitation column
    numeric_cols : list
        Numeric columns
    evaluation_results : dict
        Evaluation results
    valid_classes : list
        Valid class names
    output_dir : str
        Output directory
    """
    logger.info("\nGenerating visualizations...")
    
    # EDA plots
    plot_eda_overview(df, temp_col, precip_col, output_dir)
    plot_correlation_heatmap(df, numeric_cols, output_dir)
    
    # Cluster plots
    if 'cluster' in df.columns:
        plot_clusters(df, 'cluster', temp_col, output_dir)
    
    # Model evaluation plots
    if 'classifier' in evaluation_results:
        plot_confusion_matrix(
            evaluation_results['classifier']['confusion_matrix'],
            valid_classes,
            output_dir
        )
        
        plot_feature_importance(
            evaluation_results['classifier']['feature_importance'],
            evaluation_results['regressor']['feature_importance'],
            output_dir
        )
    
    # Model comparison - only if there are initial models
    clf_comparison = evaluation_results.get('clf_comparison', pd.DataFrame())
    reg_comparison = evaluation_results.get('reg_comparison', pd.DataFrame())
    
    if not clf_comparison.empty and not reg_comparison.empty:
        plot_model_comparison(clf_comparison, reg_comparison, output_dir)
    else:
        logger.info("⚠️ Skipping model comparison - no initial models trained")
    
    logger.info(f"All visualizations saved to '{output_dir}' directory")