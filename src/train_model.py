"""
Model training module for NHL game prediction.
Trains and evaluates different models for predicting game outcomes.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import argparse
from typing import Dict, List, Tuple, Any

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb

# Import utility functions
from utils import load_dataframe, logger, MODELS_DIR

def load_modeling_data(scaled: bool = True) -> pd.DataFrame:
    """
    Load the prepared modeling dataset.
    
    Args:
        scaled: Whether to load the scaled or unscaled version
        
    Returns:
        Modeling DataFrame
    """
    filename = 'modeling_data_scaled.csv' if scaled else 'modeling_data.csv'
    return load_dataframe(filename)

def split_train_test(df: pd.DataFrame, test_size: float = 0.2, 
                    random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                   pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Identify feature columns (excluding IDs and target)
    id_cols = ['game_id', 'date', 'home_team_id', 'away_team_id']
    target_col = 'home_win'
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in id_cols + [target_col]]
    
    # Split data
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series,
                             cv: int = 5) -> Tuple[LogisticRegression, Dict]:
    """
    Train a Logistic Regression model with grid search for hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training target
        cv: Number of cross-validation folds
        
    Returns:
        Tuple of (best model, grid search results)
    """
    logger.info("Training Logistic Regression model")
    
    # Define hyperparameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'solver': ['liblinear', 'saga', 'lbfgs'],
        'max_iter': [100, 200, 500]
    }
    
    # Create base model
    lr = LogisticRegression(random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(
        lr, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_lr = grid_search.best_estimator_
    
    logger.info(f"Best Logistic Regression parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return best_lr, grid_search.cv_results_

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                       cv: int = 5) -> Tuple[RandomForestClassifier, Dict]:
    """
    Train a Random Forest model with grid search for hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training target
        cv: Number of cross-validation folds
        
    Returns:
        Tuple of (best model, grid search results)
    """
    logger.info("Training Random Forest model")
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Create base model
    rf = RandomForestClassifier(random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(
        rf, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_rf = grid_search.best_estimator_
    
    logger.info(f"Best Random Forest parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return best_rf, grid_search.cv_results_

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                 cv: int = 5) -> Tuple[xgb.XGBClassifier, Dict]:
    """
    Train an XGBoost model with grid search for hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training target
        cv: Number of cross-validation folds
        
    Returns:
        Tuple of (best model, grid search results)
    """
    logger.info("Training XGBoost model")
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2]
    }
    
    # Create base model
    xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    # Perform grid search
    grid_search = GridSearchCV(
        xgb_model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_xgb = grid_search.best_estimator_
    
    logger.info(f"Best XGBoost parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return best_xgb, grid_search.cv_results_

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                  model_name: str) -> Dict:
    """
    Evaluate a trained model on the test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Get prediction probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Get classification report
    report = classification_report(y_test, y_pred)
    
    # Log results
    logger.info(f"\n{model_name} Evaluation Results:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Classification Report:\n{report}")
    
    # Return metrics as a dictionary
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    return metrics

def plot_feature_importance(model: Any, X_train: pd.DataFrame, model_name: str,
                           top_n: int = 20) -> None:
    """
    Plot feature importance for a trained model.
    
    Args:
        model: Trained model
        X_train: Training features
        model_name: Name of the model
        top_n: Number of top features to display
    """
    plt.figure(figsize=(12, 10))
    
    feature_names = X_train.columns
    
    # Get feature importance based on model type
    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models
        importances = np.abs(model.coef_[0])
    else:
        logger.warning(f"Cannot determine feature importance for {model_name}")
        return
    
    # Create a DataFrame for easier sorting
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Select top N features
    top_features = feature_importance.head(top_n)
    
    # Plot
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title(f'Top {top_n} Feature Importance - {model_name}')
    plt.tight_layout()
    
    # Save plot
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/{model_name}_feature_importance.png')
    plt.close()
    
    logger.info(f"Feature importance plot saved for {model_name}")

def save_model(model: Any, model_name: str) -> str:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model
        model_name: Name of the model
        
    Returns:
        Path to the saved model
    """
    # Create timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create filename
    filename = f"{model_name}_{timestamp}.joblib"
    path = os.path.join(MODELS_DIR, filename)
    
    # Save the model
    joblib.dump(model, path)
    
    logger.info(f"Model saved to {path}")
    
    return path

def compare_models(metrics_list: List[Dict]) -> None:
    """
    Compare multiple models based on their evaluation metrics.
    
    Args:
        metrics_list: List of metric dictionaries from evaluate_model
    """
    # Create a DataFrame from the metrics
    comparison_df = pd.DataFrame([
        {
            'Model': m['model_name'],
            'Accuracy': m['accuracy'],
            'Precision': m['precision'],
            'Recall': m['recall'],
            'F1 Score': m['f1'],
            'ROC AUC': m['roc_auc']
        }
        for m in metrics_list
    ])
    
    # Sort by ROC AUC
    comparison_df = comparison_df.sort_values('ROC AUC', ascending=False)
    
    # Log comparison
    logger.info("\nModel Comparison:")
    logger.info(comparison_df.to_string(index=False))
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Reshape for seaborn
    comparison_melted = pd.melt(comparison_df, id_vars=['Model'], 
                               var_name='Metric', value_name='Score')
    
    # Plot
    sns.barplot(x='Model', y='Score', hue='Metric', data=comparison_melted)
    plt.title('Model Comparison')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/model_comparison.png')
    plt.close()
    
    logger.info("Model comparison plot saved")
    
    # Also save the comparison table
    comparison_df.to_csv('figures/model_comparison.csv', index=False)

def main():
    """Main function to train models."""
    parser = argparse.ArgumentParser(description='Train NHL game prediction models')
    parser.add_argument('--scaled', action='store_true', default=True,
                        help='Use scaled features (default: True)')
    parser.add_argument('--cv', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing (default: 0.2)')
    args = parser.parse_args()
    
    logger.info("Starting model training")
    
    # Load data
    data_df = load_modeling_data(scaled=args.scaled)
    
    # Split data
    X_train, X_test, y_train, y_test = split_train_test(
        data_df, test_size=args.test_size
    )
    
    # Train models
    lr_model, lr_results = train_logistic_regression(X_train, y_train, cv=args.cv)
    rf_model, rf_results = train_random_forest(X_train, y_train, cv=args.cv)
    xgb_model, xgb_results = train_xgboost(X_train, y_train, cv=args.cv)
    
    # Evaluate models
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    
    # Plot feature importance
    plot_feature_importance(lr_model, X_train, "Logistic_Regression")
    plot_feature_importance(rf_model, X_train, "Random_Forest")
    plot_feature_importance(xgb_model, X_train, "XGBoost")
    
    # Compare models
    compare_models([lr_metrics, rf_metrics, xgb_metrics])
    
    # Save models
    save_model(lr_model, "LogisticRegression")
    save_model(rf_model, "RandomForest")
    save_model(xgb_model, "XGBoost")
    
    logger.info("Model training complete")

if __name__ == "__main__":
    main() 