"""
Prefect workflow for automated model training pipeline
Orchestrates data loading, preprocessing, training, and evaluation
"""

from prefect import flow, task
from prefect.task_runners import ConcurrentTaskRunner
import logging
from datetime import datetime
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml import config
from ml.data_loader import load_and_preprocess_data
from ml.preprocessing import (
    prepare_features_and_targets,
    create_train_test_splits,
    preprocess_train_test_data,
    perform_clustering
)
from ml.train_models import train_all_models
from ml.evaluate import evaluate_final_models
from ml.visualization import create_all_visualizations
from ml.utils import (
    save_model_artifacts,
    create_model_artifact_dict,
    save_evaluation_results,
    setup_logging,
    log_data_summary
)

# Setup logging
setup_logging(log_file="logs/prefect_training.log")
logger = logging.getLogger(__name__)


@task(name="Load and Preprocess Data", retries=2, retry_delay_seconds=10)
def load_data_task():
    """
    Task: Load and preprocess raw data
    
    Returns:
    --------
    tuple: (df, categorical_cols, numeric_cols)
    """
    logger.info("🔄 Loading and preprocessing data...")
    
    df, categorical_cols, numeric_cols = load_and_preprocess_data(
        filepath=config.DATA_PATH,
        date_column=config.DATE_COLUMN,
        missing_threshold=config.MISSING_VALUE_THRESHOLD
    )
    
    log_data_summary(df, "Preprocessed Data")
    
    return df, categorical_cols, numeric_cols


@task(name="Perform Clustering", retries=2)
def clustering_task(df, categorical_cols, numeric_cols):
    """
    Task: Perform K-Means clustering
    
    Returns:
    --------
    tuple: (df_with_clusters, cluster_metrics)
    """
    logger.info("🔄 Performing clustering analysis...")
    
    # Add humidity to cluster features if available
    cluster_features = config.CLUSTER_FEATURES.copy()
    humidity_cols = [col for col in df.columns if 'humid' in col.lower()]
    if humidity_cols:
        cluster_features.append(humidity_cols[0])
    
    cluster_labels, cluster_metrics = perform_clustering(
        df,
        cluster_features=cluster_features,
        n_clusters=config.N_CLUSTERS,
        random_state=config.RANDOM_STATE
    )
    
    df['cluster'] = cluster_labels
    
    logger.info(f"✅ Clustering complete: {cluster_metrics}")
    
    return df, cluster_metrics


@task(name="Prepare Features and Targets", retries=1)
def prepare_data_task(df):
    """
    Task: Prepare features and targets for modeling
    
    Returns:
    --------
    dict: data_dict with X, y_reg, y_clf, valid_classes
    """
    logger.info("🔄 Preparing features and targets...")
    
    data_dict = prepare_features_and_targets(
        df,
        temp_column=config.TEMPERATURE_COLUMN,
        precip_column=config.PRECIPITATION_COLUMN,
        min_samples=config.MIN_CLASS_SAMPLES
    )
    
    logger.info(f"✅ Data prepared: {len(data_dict['X'])} samples, {len(data_dict['valid_classes'])} classes")
    
    return data_dict


@task(name="Create Train-Test Splits", retries=1)
def split_data_task(data_dict):
    """
    Task: Create train-test splits
    
    Returns:
    --------
    dict: splits with train and test data
    """
    logger.info("🔄 Creating train-test splits...")
    
    splits = create_train_test_splits(
        X=data_dict['X'],
        y_reg=data_dict['y_reg'],
        y_clf=data_dict['y_clf'],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )
    
    logger.info("✅ Train-test splits created")
    
    return splits


@task(name="Preprocess Data", retries=1)
def preprocess_task(splits, categorical_cols, numeric_cols):
    """
    Task: Apply preprocessing transformations
    
    Returns:
    --------
    dict: preprocessed_data with transformed features
    """
    logger.info("🔄 Preprocessing train and test data...")
    
    preprocessed_data = preprocess_train_test_data(
        splits=splits,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        apply_smote=True
    )
    
    logger.info("✅ Preprocessing complete")
    
    return preprocessed_data


@task(name="Train Models", retries=1, timeout_seconds=600)
def train_models_task(preprocessed_data):
    """
    Task: Train all models (initial + tuned)
    
    Returns:
    --------
    dict: trained_models with best classifier and regressor
    """
    logger.info("🔄 Training models...")
    
    trained_models = train_all_models(
        preprocessed_data=preprocessed_data,
        param_grid=config.RF_PARAM_GRID,  # Remove initial_configs line
        cv=config.CV_FOLDS,
        random_state=config.RANDOM_STATE
    )
    
    logger.info("✅ All models trained successfully")
    
    return trained_models

@task(name="Evaluate Models", retries=1)
def evaluate_models_task(trained_models, preprocessed_data, valid_classes):
    """
    Task: Evaluate trained models
    
    Returns:
    --------
    dict: evaluation_results with metrics and comparisons
    """
    logger.info("🔄 Evaluating models...")
    
    evaluation_results = evaluate_final_models(
        trained_models=trained_models,
        preprocessed_data=preprocessed_data,
        valid_classes=valid_classes,
        feature_names=preprocessed_data['feature_names']
    )
    
    logger.info("✅ Model evaluation complete")
    
    return evaluation_results


@task(name="Generate Visualizations", retries=1)
def visualize_results_task(df, evaluation_results, valid_classes):
    """
    Task: Create all visualizations
    """
    logger.info("🔄 Generating visualizations...")
    
    create_all_visualizations(
        df=df,
        temp_col=config.TEMPERATURE_COLUMN,
        precip_col=config.PRECIPITATION_COLUMN,
        numeric_cols=[col for col in df.select_dtypes(include=['number']).columns],
        evaluation_results=evaluation_results,
        valid_classes=valid_classes,
        output_dir=config.PLOTS_OUTPUT_DIR
    )
    
    logger.info("✅ Visualizations created")


@task(name="Save Model Artifacts", retries=2)
def save_models_task(trained_models, preprocessed_data, evaluation_results, 
                     valid_classes, cluster_metrics):
    """
    Task: Save trained models and metadata
    
    Returns:
    --------
    str: path to saved model file
    """
    logger.info("🔄 Saving model artifacts...")
    
    artifacts = create_model_artifact_dict(
        best_classifier=trained_models['best_classifier'],
        best_regressor=trained_models['best_regressor'],
        preprocessor=preprocessed_data['preprocessor'],
        valid_classes=valid_classes,
        numeric_cols=preprocessed_data['numeric_cols'],
        categorical_cols=preprocessed_data['categorical_cols'],
        evaluation_results=evaluation_results,
        config={
            'n_clusters': config.N_CLUSTERS,
            'cluster_metrics': cluster_metrics,
            'test_size': config.TEST_SIZE,
            'random_state': config.RANDOM_STATE
        }
    )
    
    success = save_model_artifacts(artifacts, config.MODEL_OUTPUT_PATH)
    
    if success:
        logger.info(f"✅ Models saved to {config.MODEL_OUTPUT_PATH}")
        return config.MODEL_OUTPUT_PATH
    else:
        raise Exception("Failed to save model artifacts")


@task(name="Save Evaluation Results", retries=2)
def save_evaluation_task(evaluation_results, trained_models, cluster_metrics):
    """
    Task: Save evaluation results to JSON
    """
    logger.info("🔄 Saving evaluation results...")
    
    eval_summary = {
        'classifier': {
            'best_model': 'RandomForestClassifier',
            'metrics': evaluation_results['classifier']['metrics'],
            'best_params': trained_models['clf_tuning_results']['best_params']
        },
        'regressor': {
            'best_model': 'RandomForestRegressor',
            'metrics': evaluation_results['regressor']['metrics'],
            'best_params': trained_models['reg_tuning_results']['best_params']
        },
        'clustering': cluster_metrics,
        'initial_model_comparison': {
            'classification': evaluation_results['clf_comparison'].to_dict(),
            'regression': evaluation_results['reg_comparison'].to_dict()
        }
    }
    
    success = save_evaluation_results(eval_summary, "models/evaluation_results.json")
    
    if success:
        logger.info("✅ Evaluation results saved")
    else:
        logger.warning("⚠️ Failed to save evaluation results")


@task(name="Send Success Notification")
def send_success_notification(evaluation_results):
    """
    Task: Send notification on successful completion
    """
    from workflows.notifications import send_discord_webhook
    
    clf_acc = evaluation_results['classifier']['metrics']['accuracy']
    clf_f1 = evaluation_results['classifier']['metrics']['f1_macro']
    reg_r2 = evaluation_results['regressor']['metrics']['r2']
    reg_rmse = evaluation_results['regressor']['metrics']['rmse']
    
    message = f"""
✅ **MODEL TRAINING COMPLETED SUCCESSFULLY!**

⏰ **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 **Classification Results:**
   • Accuracy: {clf_acc:.4f} ({clf_acc*100:.2f}%)
   • F1-Score: {clf_f1:.4f}

📈 **Regression Results:**
   • R² Score: {reg_r2:.4f}
   • RMSE: {reg_rmse:.4f}°C

💾 **Models saved to:** `{config.MODEL_OUTPUT_PATH}`
    """
    
    logger.info(message)
    print(message)
    
    # Send Discord notification
    discord_sent = send_discord_webhook(message)
    if discord_sent:
        logger.info("✅ Discord notification sent successfully")
    else:
        logger.warning("⚠️ Failed to send Discord notification (check DISCORD_WEBHOOK_URL)")


@task(name="Send Failure Notification")
def send_failure_notification(error_message):
    """
    Task: Send notification on failure
    """
    from workflows.notifications import send_discord_webhook
    
    message = f"""
❌ **MODEL TRAINING FAILED!**

⏰ **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

⚠️ **Error:** 
```
{error_message}
```

📝 **Check logs at:** `logs/prefect_training.log`
    """
    
    logger.error(message)
    print(message)
    
    # Send Discord notification
    discord_sent = send_discord_webhook(message)
    if discord_sent:
        logger.info("✅ Discord failure notification sent")
    else:
        logger.warning("⚠️ Failed to send Discord notification")


@flow(
    name="Weather Prediction Model Training Pipeline",
    task_runner=ConcurrentTaskRunner(),
    log_prints=True
)
def training_pipeline():
    """
    Main Prefect flow for complete model training pipeline
    
    This flow orchestrates:
    1. Data loading and preprocessing
    2. Clustering analysis
    3. Feature preparation
    4. Train-test splitting
    5. Data preprocessing with SMOTE
    6. Model training (classification, regression)
    7. Hyperparameter tuning
    8. Model evaluation
    9. Visualization generation
    10. Model and results saving
    11. Notifications
    """
    try:
        logger.info("="*70)
        logger.info("PREFECT: Weather Prediction Training Pipeline Started")
        logger.info("="*70)
        
        # Step 1: Load data
        df, categorical_cols, numeric_cols = load_data_task()
        
        # Step 2: Clustering (can run independently)
        df_clustered, cluster_metrics = clustering_task(df, categorical_cols, numeric_cols)
        
        # Step 3: Prepare data
        data_dict = prepare_data_task(df_clustered)
        
        # Step 4: Split data
        splits = split_data_task(data_dict)
        
        # Step 5: Preprocess
        preprocessed_data = preprocess_task(splits, categorical_cols, numeric_cols)
        
        # Step 6: Train models
        trained_models = train_models_task(preprocessed_data)
        
        # Step 7: Evaluate models
        evaluation_results = evaluate_models_task(
            trained_models, 
            preprocessed_data, 
            data_dict['valid_classes']
        )
        
        # Step 8: Generate visualizations
        visualize_results_task(df_clustered, evaluation_results, data_dict['valid_classes'])
        
        # Step 9: Save models
        model_path = save_models_task(
            trained_models,
            preprocessed_data,
            evaluation_results,
            data_dict['valid_classes'],
            cluster_metrics
        )
        
        # Step 10: Save evaluation results
        save_evaluation_task(evaluation_results, trained_models, cluster_metrics)
        
        # Step 11: Send success notification
        send_success_notification(evaluation_results)
        
        logger.info("="*70)
        logger.info("✅ PREFECT: Pipeline Completed Successfully!")
        logger.info("="*70)
        
        return {
            'status': 'success',
            'model_path': model_path,
            'metrics': {
                'classifier_accuracy': evaluation_results['classifier']['metrics']['accuracy'],
                'regressor_r2': evaluation_results['regressor']['metrics']['r2']
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}", exc_info=True)
        send_failure_notification(str(e))
        raise


if __name__ == "__main__":
    # Run the flow
    result = training_pipeline()
    print(f"\n🎉 Pipeline result: {result}")