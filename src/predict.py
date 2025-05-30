"""
Prediction module for NHL game prediction.
Loads trained models and generates predictions for upcoming games.
"""
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import argparse
import glob
from typing import Dict, List, Tuple, Union, Any

# Import utility functions
from utils import (
    make_api_request, get_team_data, get_season_dates,
    save_dataframe, load_dataframe, logger, MODELS_DIR
)

def get_latest_model(model_name: str) -> str:
    """
    Get the path to the latest trained model of a specific type.
    
    Args:
        model_name: Name of the model (e.g., 'LogisticRegression', 'RandomForest', 'XGBoost')
        
    Returns:
        Path to the latest model file
    """
    # Find all model files matching the pattern
    model_pattern = os.path.join(MODELS_DIR, f"{model_name}_*.joblib")
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        raise FileNotFoundError(f"No models found matching pattern: {model_pattern}")
    
    # Sort by modification time (newest first)
    latest_model = max(model_files, key=os.path.getmtime)
    
    logger.info(f"Using model: {latest_model}")
    
    return latest_model

def load_model(model_path: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded model
    """
    # Load the model
    model = joblib.load(model_path)
    
    return model

def get_upcoming_games(days_ahead: int = 7) -> pd.DataFrame:
    """
    Get upcoming NHL games for the specified number of days ahead.
    
    Args:
        days_ahead: Number of days to look ahead
        
    Returns:
        DataFrame with upcoming games
    """
    # Get today's date
    today = datetime.now()
    
    # Create date range
    date_range = [
        (today + timedelta(days=i)).strftime('%Y-%m-%d')
        for i in range(days_ahead)
    ]
    
    all_games = []
    
    # Fetch games for each date
    for date in date_range:
        try:
            logger.info(f"Fetching games for {date}")
            
            # Make API request
            response = make_api_request(f"schedule?date={date}")
            
            # Extract games
            for date_info in response.get('dates', []):
                for game in date_info.get('games', []):
                    game_data = {
                        'game_id': game.get('gamePk'),
                        'date': date_info.get('date'),
                        'home_team_id': game.get('teams', {}).get('home', {}).get('team', {}).get('id'),
                        'away_team_id': game.get('teams', {}).get('away', {}).get('team', {}).get('id'),
                        'venue': game.get('venue', {}).get('name'),
                        'status': game.get('status', {}).get('abstractGameState')
                    }
                    all_games.append(game_data)
        except Exception as e:
            logger.error(f"Error fetching games for {date}: {e}")
    
    # Create DataFrame
    if not all_games:
        logger.warning(f"No upcoming games found for the next {days_ahead} days")
        return pd.DataFrame()
    
    games_df = pd.DataFrame(all_games)
    
    # Add team names
    teams_df = get_team_data()
    
    games_df = pd.merge(
        games_df,
        teams_df[['team_id', 'team_name']],
        left_on='home_team_id',
        right_on='team_id',
        how='left'
    ).rename(columns={'team_name': 'home_team'})
    
    games_df = pd.merge(
        games_df,
        teams_df[['team_id', 'team_name']],
        left_on='away_team_id',
        right_on='team_id',
        how='left'
    ).rename(columns={'team_name': 'away_team'})
    
    # Drop redundant columns and filter only scheduled games
    games_df = games_df.drop(columns=['team_id_x', 'team_id_y'], errors='ignore')
    games_df = games_df[games_df['status'] == 'Preview']
    
    logger.info(f"Found {len(games_df)} upcoming games")
    
    return games_df

def prepare_features_for_prediction(upcoming_games: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for upcoming games by fetching current team statistics.
    
    Args:
        upcoming_games: DataFrame with upcoming games
        
    Returns:
        DataFrame with prepared features
    """
    # Create a copy
    games = upcoming_games.copy()
    
    # Determine current season
    current_year = datetime.now().year
    month = datetime.now().month
    
    # If it's between January and August, we're in the previous season
    if 1 <= month <= 8:
        season = f"{current_year-1}{current_year}"
    else:
        season = f"{current_year}{current_year+1}"
    
    logger.info(f"Using season {season} for predictions")
    
    # Load team stats
    try:
        team_stats_df = load_dataframe('team_stats_clean.csv')
        team_features_df = load_dataframe('team_features.csv')
        aggregated_player_stats = load_dataframe('aggregated_player_stats.csv')
        
        # Filter to current season if available
        current_team_stats = team_stats_df[team_stats_df['season'] == season]
        current_team_features = team_features_df[team_features_df['season'] == season]
        current_player_stats = aggregated_player_stats[aggregated_player_stats['season'] == season]
        
        # If no data for current season, use most recent
        if current_team_stats.empty:
            logger.warning(f"No team stats found for season {season}. Using most recent data.")
            current_team_stats = team_stats_df.sort_values('season', ascending=False)
            current_team_features = team_features_df.sort_values('season', ascending=False)
            current_player_stats = aggregated_player_stats.sort_values('season', ascending=False)
    except FileNotFoundError:
        logger.warning("Preprocessed data not found. Fetching current stats from API.")
        
        # Fetch current team stats from API
        from data_collection import fetch_team_stats, fetch_player_stats
        from preprocessing import clean_team_stats, clean_player_stats, extract_team_features, aggregate_player_stats_by_team
        
        current_team_stats = fetch_team_stats(season)
        current_team_stats = clean_team_stats(current_team_stats)
        current_team_features = extract_team_features(current_team_stats)
        
        current_player_stats = fetch_player_stats(season)
        current_player_stats = clean_player_stats(current_player_stats)
        current_player_stats = aggregate_player_stats_by_team(current_player_stats)
    
    # Add season to games
    games['season'] = season
    
    # Add team stats for home and away teams
    games_with_stats = pd.merge(
        games, 
        current_team_features, 
        left_on=['home_team_id'], 
        right_on=['team_id'], 
        how='left',
        suffixes=('', '_home')
    )
    
    games_with_stats = pd.merge(
        games_with_stats, 
        current_team_features, 
        left_on=['away_team_id'], 
        right_on=['team_id'], 
        how='left',
        suffixes=('_home', '_away')
    )
    
    # Add player stats
    games_with_stats = pd.merge(
        games_with_stats, 
        current_player_stats, 
        left_on=['home_team_id'], 
        right_on=['team_id'], 
        how='left',
        suffixes=('', '_home_players')
    )
    
    games_with_stats = pd.merge(
        games_with_stats, 
        current_player_stats, 
        left_on=['away_team_id'], 
        right_on=['team_id'], 
        how='left',
        suffixes=('_home_players', '_away_players')
    )
    
    # Drop duplicate columns
    cols_to_drop = [col for col in games_with_stats.columns 
                   if col.endswith('_x') or col.endswith('_y')
                   or col.startswith('team_id')]
    games_with_stats = games_with_stats.drop(columns=cols_to_drop, errors='ignore')
    
    # Create relative features (home vs away)
    relative_feature_pairs = [
        ('win_pct_home', 'win_pct_away'),
        ('goals_per_game_home', 'goals_per_game_away'),
        ('goals_against_per_game_home', 'goals_against_per_game_away'),
        ('powerplay_pct_home', 'powerplay_pct_away'),
        ('penalty_kill_pct_home', 'penalty_kill_pct_away'),
        ('shots_per_game_home', 'shots_per_game_away'),
        ('shots_allowed_home', 'shots_allowed_away')
    ]
    
    for home_feat, away_feat in relative_feature_pairs:
        if home_feat in games_with_stats.columns and away_feat in games_with_stats.columns:
            games_with_stats[f'rel_{home_feat.replace("_home", "")}'] = (
                games_with_stats[home_feat] - games_with_stats[away_feat]
            )
    
    # Fill missing values with 0
    games_with_stats = games_with_stats.fillna(0)
    
    logger.info(f"Prepared features for {len(games_with_stats)} games")
    
    return games_with_stats

def align_features_with_model(games_with_features: pd.DataFrame, model_features: List[str]) -> pd.DataFrame:
    """
    Align features in the prediction dataset with those used by the model.
    
    Args:
        games_with_features: DataFrame with game features
        model_features: List of feature names used by the model
        
    Returns:
        DataFrame with aligned features
    """
    # Create a copy
    df = games_with_features.copy()
    
    # Check for missing features
    missing_features = [feat for feat in model_features if feat not in df.columns]
    extra_features = [feat for feat in df.columns if feat not in model_features and feat not in [
        'game_id', 'date', 'home_team_id', 'away_team_id', 'venue', 'status',
        'home_team', 'away_team', 'season'
    ]]
    
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
        
        # Add missing features with zeros
        for feat in missing_features:
            df[feat] = 0
    
    # Select only necessary columns for prediction
    prediction_df = df[model_features]
    
    logger.info(f"Aligned features: {prediction_df.shape[1]} features")
    
    return prediction_df

def predict_games(upcoming_games: pd.DataFrame, model: Any, model_features: List[str]) -> pd.DataFrame:
    """
    Generate predictions for upcoming games.
    
    Args:
        upcoming_games: DataFrame with upcoming games and features
        model: Trained model
        model_features: Features used by the model
        
    Returns:
        DataFrame with predictions
    """
    # Create a copy
    games = upcoming_games.copy()
    
    # Align features
    prediction_features = align_features_with_model(games, model_features)
    
    # Make predictions
    y_pred_proba = model.predict_proba(prediction_features)
    
    # Add predictions to the DataFrame
    games['home_win_probability'] = y_pred_proba[:, 1]
    games['away_win_probability'] = y_pred_proba[:, 0]
    
    # Determine predicted winner
    games['predicted_winner'] = np.where(
        games['home_win_probability'] > 0.5,
        games['home_team'],
        games['away_team']
    )
    
    games['win_probability'] = np.where(
        games['home_win_probability'] > 0.5,
        games['home_win_probability'],
        games['away_win_probability']
    )
    
    logger.info(f"Generated predictions for {len(games)} games")
    
    return games

def format_prediction_output(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Format prediction outputs for display/saving.
    
    Args:
        predictions: DataFrame with raw predictions
        
    Returns:
        Formatted DataFrame
    """
    # Create a copy
    df = predictions.copy()
    
    # Select and reorder columns
    display_cols = [
        'date', 'home_team', 'away_team', 'predicted_winner',
        'win_probability', 'home_win_probability', 'away_win_probability'
    ]
    
    formatted_df = df[display_cols].copy()
    
    # Format probabilities as percentages
    probability_cols = ['win_probability', 'home_win_probability', 'away_win_probability']
    for col in probability_cols:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2%}")
    
    # Sort by date
    formatted_df = formatted_df.sort_values('date')
    
    return formatted_df

def main():
    """Main function to generate predictions."""
    parser = argparse.ArgumentParser(description='Generate NHL game predictions')
    parser.add_argument('--days_ahead', type=int, default=7,
                        help='Number of days ahead to predict (default: 7)')
    parser.add_argument('--model', type=str, default='XGBoost',
                        choices=['LogisticRegression', 'RandomForest', 'XGBoost'],
                        help='Model to use for predictions (default: XGBoost)')
    args = parser.parse_args()
    
    logger.info("Starting prediction generation")
    
    # Get upcoming games
    upcoming_games = get_upcoming_games(days_ahead=args.days_ahead)
    
    if upcoming_games.empty:
        logger.warning("No upcoming games found. Exiting.")
        return
    
    # Prepare features
    games_with_features = prepare_features_for_prediction(upcoming_games)
    
    # Load model
    model_path = get_latest_model(args.model)
    model = load_model(model_path)
    
    # Get model features
    if hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_
    else:
        # Try to load features from a processed dataset
        try:
            modeling_df = load_dataframe('modeling_data_scaled.csv')
            id_cols = ['game_id', 'date', 'home_team_id', 'away_team_id']
            target_col = 'home_win'
            model_features = [col for col in modeling_df.columns 
                             if col not in id_cols + [target_col]]
        except:
            logger.error("Cannot determine model features. Exiting.")
            return
    
    # Generate predictions
    predictions = predict_games(games_with_features, model, model_features)
    
    # Format predictions for display
    formatted_predictions = format_prediction_output(predictions)
    
    # Display predictions
    logger.info("\nPredictions for upcoming games:")
    logger.info(formatted_predictions.to_string(index=False))
    
    # Save predictions
    os.makedirs('predictions', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    predictions_path = f"predictions/game_predictions_{timestamp}.csv"
    formatted_predictions.to_csv(predictions_path, index=False)
    
    logger.info(f"Predictions saved to {predictions_path}")

if __name__ == "__main__":
    main() 