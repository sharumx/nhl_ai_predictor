"""
Data preprocessing module for NHL game prediction model.
Handles cleaning, feature engineering, and preparation of data for modeling.
"""
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler

# Import utility functions
from src.utils import (
    load_dataframe, save_dataframe, encode_categorical,
    calculate_recent_form, logger, PROCESSED_DATA_DIR
)

def preprocess_schedule(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess schedule data.
    
    Args:
        schedule_df: Raw schedule DataFrame
        
    Returns:
        Preprocessed schedule DataFrame
    """
    logger.info("Preprocessing schedule data")
    
    if schedule_df.empty:
        logger.warning("Empty schedule DataFrame provided")
        return schedule_df
    
    # Ensure proper data types
    schedule_df['date'] = pd.to_datetime(schedule_df['date'])
    schedule_df['game_id'] = schedule_df['game_id'].astype('int64')
    schedule_df['home_team_id'] = schedule_df['home_team_id'].astype('int64')
    schedule_df['away_team_id'] = schedule_df['away_team_id'].astype('int64')
    
    # Convert scores to numeric, handling missing values
    schedule_df['home_score'] = pd.to_numeric(schedule_df['home_score'], errors='coerce').fillna(0).astype('int64')
    schedule_df['away_score'] = pd.to_numeric(schedule_df['away_score'], errors='coerce').fillna(0).astype('int64')
    
    # Add derived features
    schedule_df['total_score'] = schedule_df['home_score'] + schedule_df['away_score']
    schedule_df['score_diff'] = schedule_df['home_score'] - schedule_df['away_score']
    
    # Determine winner (1 for home, 0 for away, 0.5 for tie)
    schedule_df['home_win'] = np.where(
        schedule_df['home_score'] > schedule_df['away_score'], 1,
        np.where(schedule_df['home_score'] < schedule_df['away_score'], 0, 0.5)
    )
    
    # Only include completed games
    completed_games = schedule_df[schedule_df['status'] == 'Final'].copy()
    
    return completed_games

def preprocess_game_stats(game_stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess game statistics data.
    
    Args:
        game_stats_df: Raw game stats DataFrame
        
    Returns:
        Preprocessed game stats DataFrame
    """
    logger.info("Preprocessing game stats data")
    
    if game_stats_df.empty:
        logger.warning("Empty game stats DataFrame provided")
        return game_stats_df
    
    # Ensure proper data types
    game_stats_df['game_id'] = game_stats_df['game_id'].astype('int64')
    
    numeric_cols = [col for col in game_stats_df.columns if col != 'game_id']
    
    # Convert all numeric columns and handle missing values
    for col in numeric_cols:
        game_stats_df[col] = pd.to_numeric(game_stats_df[col], errors='coerce').fillna(0)
    
    # Calculate additional metrics
    # Home/away shot efficiency
    game_stats_df['home_shot_pct'] = np.where(
        game_stats_df['home_shots'] > 0,
        game_stats_df['home_goals'] / game_stats_df['home_shots'], 
        0
    )
    
    game_stats_df['away_shot_pct'] = np.where(
        game_stats_df['away_shots'] > 0,
        game_stats_df['away_goals'] / game_stats_df['away_shots'],
        0
    )
    
    # Home/away power play efficiency
    game_stats_df['home_pp_pct'] = np.where(
        game_stats_df['home_powerplay_opportunities'] > 0,
        game_stats_df['home_powerplay_goals'] / game_stats_df['home_powerplay_opportunities'],
        0
    )
    
    game_stats_df['away_pp_pct'] = np.where(
        game_stats_df['away_powerplay_opportunities'] > 0,
        game_stats_df['away_powerplay_goals'] / game_stats_df['away_powerplay_opportunities'],
        0
    )
    
    return game_stats_df

def preprocess_team_stats(team_stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess team statistics data.
    
    Args:
        team_stats_df: Raw team stats DataFrame
        
    Returns:
        Preprocessed team stats DataFrame
    """
    logger.info("Preprocessing team stats data")
    
    if team_stats_df.empty:
        logger.warning("Empty team stats DataFrame provided")
        return team_stats_df
    
    # Ensure proper data types
    team_stats_df['team_id'] = team_stats_df['team_id'].astype('int64')
    
    # Convert all numeric columns except team_id and season
    numeric_cols = [col for col in team_stats_df.columns if col not in ['team_id', 'season']]
    
    for col in numeric_cols:
        team_stats_df[col] = pd.to_numeric(team_stats_df[col], errors='coerce').fillna(0)
    
    # Calculate additional team metrics
    team_stats_df['win_pct'] = np.where(
        team_stats_df['games_played'] > 0,
        team_stats_df['wins'] / team_stats_df['games_played'],
        0
    )
    
    team_stats_df['points_pct'] = np.where(
        team_stats_df['games_played'] > 0,
        team_stats_df['pts'] / (team_stats_df['games_played'] * 2),  # 2 points per game
        0
    )
    
    return team_stats_df

def preprocess_player_stats(player_stats_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess player statistics data.
    
    Args:
        player_stats_df: Raw player stats DataFrame
        
    Returns:
        Tuple of (skaters_df, goalies_df)
    """
    logger.info("Preprocessing player stats data")
    
    if player_stats_df.empty:
        logger.warning("Empty player stats DataFrame provided")
        # Return empty dataframes with expected columns
        skaters_cols = ['player_id', 'player_name', 'team_id', 'position', 'position_type', 
                         'position_code', 'season', 'games', 'goals', 'assists', 'points', 
                         'plus_minus', 'points_per_game']
        goalies_cols = ['player_id', 'player_name', 'team_id', 'position', 'position_type', 
                         'position_code', 'season', 'games', 'wins', 'losses', 'save_pct', 
                         'goals_against_avg', 'win_pct']
        return pd.DataFrame(columns=skaters_cols), pd.DataFrame(columns=goalies_cols)
    
    # Convert data types
    player_stats_df['player_id'] = player_stats_df['player_id'].astype('int64')
    player_stats_df['team_id'] = player_stats_df['team_id'].astype('int64')
    
    # Split into skaters and goalies
    skaters_df = player_stats_df[player_stats_df['position_type'] != 'Goalie'].copy()
    goalies_df = player_stats_df[player_stats_df['position_type'] == 'Goalie'].copy()
    
    # Process skaters
    skater_numeric_cols = ['games', 'goals', 'assists', 'points', 'plus_minus']
    
    for col in skater_numeric_cols:
        if col in skaters_df.columns:
            skaters_df[col] = pd.to_numeric(skaters_df[col], errors='coerce').fillna(0)
    
    # Calculate points per game
    skaters_df['points_per_game'] = np.where(
        skaters_df['games'] > 0,
        skaters_df['points'] / skaters_df['games'],
        0
    )
    
    # Process goalies
    goalie_numeric_cols = ['games', 'wins', 'losses', 'save_pct', 'goals_against_avg']
    
    for col in goalie_numeric_cols:
        if col in goalies_df.columns:
            goalies_df[col] = pd.to_numeric(goalies_df[col], errors='coerce').fillna(0)
    
    # Calculate win percentage
    goalies_df['win_pct'] = np.where(
        goalies_df['games'] > 0,
        goalies_df['wins'] / goalies_df['games'],
        0
    )
    
    return skaters_df, goalies_df

def prepare_model_data(
    schedule_df: pd.DataFrame,
    team_stats_df: pd.DataFrame,
    game_stats_df: pd.DataFrame,
    recent_games: int = 10
) -> pd.DataFrame:
    """
    Prepare data for model training.
    
    Args:
        schedule_df: Preprocessed schedule DataFrame
        team_stats_df: Preprocessed team stats DataFrame
        game_stats_df: Preprocessed game stats DataFrame
        recent_games: Number of recent games to include for form calculation
        
    Returns:
        DataFrame ready for model training
    """
    logger.info("Preparing model data")
    
    # Check if we have data to work with
    if schedule_df.empty or team_stats_df.empty or game_stats_df.empty:
        logger.warning("One or more input DataFrames are empty")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            'game_id', 'date', 'home_team_id', 'away_team_id', 'home_win', 'home_score', 'away_score'
        ])
    
    # 1. Start with completed games from schedule
    model_data = schedule_df.copy()
    
    # Ensure date is in datetime format and sort
    model_data['date'] = pd.to_datetime(model_data['date'])
    model_data = model_data.sort_values('date')
    
    # 2. Add team stats from the corresponding season
    seasons = model_data['season'].unique()
    
    # Keep track of team form per season
    team_form_data = {}
    
    for season in seasons:
        season_mask = model_data['season'] == season
        season_games = model_data[season_mask].copy()
        
        # Get team stats for this season
        season_team_stats = team_stats_df[team_stats_df['season'] == season].copy()
        
        if not season_team_stats.empty:
            # Add home team stats
            home_stats = season_team_stats.copy()
            home_stats.columns = ['home_' + col if col not in ['team_id', 'season'] else col for col in home_stats.columns]
            model_data = pd.merge(
                model_data, 
                home_stats, 
                left_on=['home_team_id', 'season'], 
                right_on=['team_id', 'season'], 
                how='left'
            )
            model_data.drop('team_id', axis=1, inplace=True)
            
            # Add away team stats
            away_stats = season_team_stats.copy()
            away_stats.columns = ['away_' + col if col not in ['team_id', 'season'] else col for col in away_stats.columns]
            model_data = pd.merge(
                model_data, 
                away_stats, 
                left_on=['away_team_id', 'season'], 
                right_on=['team_id', 'season'], 
                how='left'
            )
            model_data.drop('team_id', axis=1, inplace=True)
        
        # Calculate team form based on recent games
        for team_id in list(set(season_games['home_team_id'].tolist() + season_games['away_team_id'].tolist())):
            team_games = []
            
            # Get home games
            home_games = season_games[season_games['home_team_id'] == team_id].copy()
            home_games['is_home'] = 1
            home_games['team_score'] = home_games['home_score']
            home_games['opponent_score'] = home_games['away_score']
            home_games['opponent_id'] = home_games['away_team_id']
            home_games['win'] = home_games['home_win']
            
            # Get away games
            away_games = season_games[season_games['away_team_id'] == team_id].copy()
            away_games['is_home'] = 0
            away_games['team_score'] = away_games['away_score']
            away_games['opponent_score'] = away_games['home_score']
            away_games['opponent_id'] = away_games['home_team_id']
            away_games['win'] = 1 - away_games['home_win']  # Invert home_win
            
            # Combine and sort by date
            team_games = pd.concat([home_games, away_games])
            team_games = team_games.sort_values('date')
            
            # Calculate rolling stats
            if not team_games.empty:
                team_games['last_n_wins'] = team_games['win'].rolling(window=recent_games, min_periods=1).sum()
                team_games['last_n_goals_scored'] = team_games['team_score'].rolling(window=recent_games, min_periods=1).mean()
                team_games['last_n_goals_conceded'] = team_games['opponent_score'].rolling(window=recent_games, min_periods=1).mean()
                
                # Store in dictionary for later merging
                team_form_data[(season, team_id)] = team_games[['game_id', 'last_n_wins', 'last_n_goals_scored', 'last_n_goals_conceded']]
    
    # 3. Merge team form data back to model data
    for (season, team_id), form_df in team_form_data.items():
        # For home team
        home_mask = (model_data['season'] == season) & (model_data['home_team_id'] == team_id)
        model_data.loc[home_mask, 'home_last_n_wins'] = form_df['last_n_wins']
        model_data.loc[home_mask, 'home_last_n_goals_scored'] = form_df['last_n_goals_scored']
        model_data.loc[home_mask, 'home_last_n_goals_conceded'] = form_df['last_n_goals_conceded']
        
        # For away team
        away_mask = (model_data['season'] == season) & (model_data['away_team_id'] == team_id)
        model_data.loc[away_mask, 'away_last_n_wins'] = form_df['last_n_wins']
        model_data.loc[away_mask, 'away_last_n_goals_scored'] = form_df['last_n_goals_scored']
        model_data.loc[away_mask, 'away_last_n_goals_conceded'] = form_df['last_n_goals_conceded']
    
    # 4. Add game stats
    model_data = pd.merge(model_data, game_stats_df, on='game_id', how='left')
    
    # 5. Fill NA values
    numeric_cols = model_data.select_dtypes(include=[np.number]).columns
    model_data[numeric_cols] = model_data[numeric_cols].fillna(0)
    
    # 6. Drop unnecessary columns
    cols_to_drop = ['venue', 'status']
    model_data = model_data.drop([col for col in cols_to_drop if col in model_data.columns], axis=1)
    
    # 7. One-hot encode categorical features (conference, division)
    categorical_cols = [col for col in model_data.columns if 'conference' in col or 'division' in col]
    model_data = encode_categorical(model_data, categorical_cols)
    
    return model_data

def prepare_prediction_data(
    upcoming_games: pd.DataFrame,
    team_stats_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    recent_games: int = 10
) -> pd.DataFrame:
    """
    Prepare data for prediction.
    
    Args:
        upcoming_games: DataFrame of upcoming games
        team_stats_df: Preprocessed team stats DataFrame
        schedule_df: Preprocessed historical schedule DataFrame
        recent_games: Number of recent games to include for form calculation
        
    Returns:
        DataFrame ready for prediction
    """
    logger.info("Preparing prediction data")
    
    if upcoming_games.empty:
        logger.warning("No upcoming games provided")
        return pd.DataFrame()
    
    # Start with upcoming games
    pred_data = upcoming_games.copy()
    
    # Ensure date is datetime
    pred_data['date'] = pd.to_datetime(pred_data['date'])
    
    # Get the most recent season from the data
    current_season = team_stats_df['season'].max()
    
    if pd.isna(current_season) or current_season == '':
        logger.warning("No valid season found in team stats, using first season from upcoming games")
        current_season = pred_data['season'].iloc[0] if 'season' in pred_data.columns else '20222023'
    
    # Ensure season is set
    if 'season' not in pred_data.columns:
        pred_data['season'] = current_season
    
    # Get team stats for the current season
    season_team_stats = team_stats_df[team_stats_df['season'] == current_season].copy()
    
    if not season_team_stats.empty:
        # Add home team stats
        home_stats = season_team_stats.copy()
        home_stats.columns = ['home_' + col if col not in ['team_id', 'season'] else col for col in home_stats.columns]
        pred_data = pd.merge(
            pred_data, 
            home_stats, 
            left_on=['home_team_id', 'season'], 
            right_on=['team_id', 'season'], 
            how='left'
        )
        pred_data.drop('team_id', axis=1, inplace=True)
        
        # Add away team stats
        away_stats = season_team_stats.copy()
        away_stats.columns = ['away_' + col if col not in ['team_id', 'season'] else col for col in away_stats.columns]
        pred_data = pd.merge(
            pred_data, 
            away_stats, 
            left_on=['away_team_id', 'season'], 
            right_on=['team_id', 'season'], 
            how='left'
        )
        pred_data.drop('team_id', axis=1, inplace=True)
    
    # Calculate team form based on recent completed games
    completed_games = schedule_df[schedule_df['status'] == 'Final'].copy()
    completed_games['date'] = pd.to_datetime(completed_games['date'])
    
    for team_id in list(set(pred_data['home_team_id'].tolist() + pred_data['away_team_id'].tolist())):
        team_games = []
        
        # Get home games
        home_games = completed_games[completed_games['home_team_id'] == team_id].copy()
        home_games['is_home'] = 1
        home_games['team_score'] = home_games['home_score']
        home_games['opponent_score'] = home_games['away_score']
        home_games['opponent_id'] = home_games['away_team_id']
        home_games['win'] = home_games['home_win']
        
        # Get away games
        away_games = completed_games[completed_games['away_team_id'] == team_id].copy()
        away_games['is_home'] = 0
        away_games['team_score'] = away_games['away_score']
        away_games['opponent_score'] = away_games['home_score']
        away_games['opponent_id'] = away_games['home_team_id']
        away_games['win'] = 1 - away_games['home_win']  # Invert home_win
        
        # Combine and sort by date
        team_games = pd.concat([home_games, away_games])
        team_games = team_games.sort_values('date')
        
        # Get the last N games
        recent_team_games = team_games.tail(recent_games)
        
        if not recent_team_games.empty:
            # Calculate stats
            last_n_wins = recent_team_games['win'].sum()
            last_n_goals_scored = recent_team_games['team_score'].mean()
            last_n_goals_conceded = recent_team_games['opponent_score'].mean()
            
            # Apply to home teams
            home_mask = pred_data['home_team_id'] == team_id
            pred_data.loc[home_mask, 'home_last_n_wins'] = last_n_wins
            pred_data.loc[home_mask, 'home_last_n_goals_scored'] = last_n_goals_scored
            pred_data.loc[home_mask, 'home_last_n_goals_conceded'] = last_n_goals_conceded
            
            # Apply to away teams
            away_mask = pred_data['away_team_id'] == team_id
            pred_data.loc[away_mask, 'away_last_n_wins'] = last_n_wins
            pred_data.loc[away_mask, 'away_last_n_goals_scored'] = last_n_goals_scored
            pred_data.loc[away_mask, 'away_last_n_goals_conceded'] = last_n_goals_conceded
    
    # Fill NA values
    numeric_cols = pred_data.select_dtypes(include=[np.number]).columns
    pred_data[numeric_cols] = pred_data[numeric_cols].fillna(0)
    
    # Drop unnecessary columns
    cols_to_drop = ['venue', 'status', 'home_score', 'away_score', 'total_score', 'score_diff', 'home_win']
    pred_data = pred_data.drop([col for col in cols_to_drop if col in pred_data.columns], axis=1)
    
    # One-hot encode categorical features (conference, division)
    categorical_cols = [col for col in pred_data.columns if 'conference' in col or 'division' in col]
    pred_data = encode_categorical(pred_data, categorical_cols)
    
    return pred_data

def preprocess_data(recent_games: int = 10) -> Dict[str, pd.DataFrame]:
    """
    Preprocess all collected data and prepare for modeling.
    
    Args:
        recent_games: Number of recent games to include for form calculation
        
    Returns:
        Dictionary of processed DataFrames
    """
    logger.info("Starting data preprocessing")
    
    try:
        # Load raw data
        try:
            schedule_df = load_dataframe('schedule.csv', processed=False)
        except FileNotFoundError:
            logger.error("Schedule data file not found. Using sample data.")
            from src.utils import create_sample_dataframe
            schedule_df = create_sample_dataframe('schedule.csv')
        
        try:
            game_stats_df = load_dataframe('game_stats.csv', processed=False)
        except FileNotFoundError:
            logger.error("Game stats data file not found. Using sample data.")
            from src.utils import create_sample_dataframe
            game_stats_df = create_sample_dataframe('game_stats.csv')
        
        try:
            team_stats_df = load_dataframe('team_stats.csv', processed=False)
        except FileNotFoundError:
            logger.error("Team stats data file not found. Using sample data.")
            from src.utils import create_sample_dataframe
            team_stats_df = create_sample_dataframe('team_stats.csv')
        
        try:
            player_stats_df = load_dataframe('player_stats.csv', processed=False)
        except FileNotFoundError:
            logger.error("Player stats data file not found. Using sample data.")
            from src.utils import create_sample_dataframe
            player_stats_df = create_sample_dataframe('player_stats.csv')
        
        # Preprocess individual datasets
        processed_schedule = preprocess_schedule(schedule_df)
        processed_game_stats = preprocess_game_stats(game_stats_df)
        processed_team_stats = preprocess_team_stats(team_stats_df)
        processed_skaters, processed_goalies = preprocess_player_stats(player_stats_df)
        
        # Prepare model data
        model_data = prepare_model_data(
            processed_schedule,
            processed_team_stats,
            processed_game_stats,
            recent_games
        )
        
        # Save processed data
        processed_data = {
            'schedule': processed_schedule,
            'game_stats': processed_game_stats,
            'team_stats': processed_team_stats,
            'skaters': processed_skaters,
            'goalies': processed_goalies,
            'model_data': model_data
        }
        
        for name, df in processed_data.items():
            save_dataframe(df, f"{name}.csv")
        
        return processed_data
    
    except Exception as e:
        logger.error(f"Error in preprocessing data: {e}")
        
        # Create and return minimal processed data
        from src.utils import create_sample_dataframe
        
        processed_schedule = preprocess_schedule(create_sample_dataframe('schedule.csv'))
        processed_game_stats = preprocess_game_stats(create_sample_dataframe('game_stats.csv'))
        processed_team_stats = preprocess_team_stats(create_sample_dataframe('team_stats.csv'))
        processed_skaters, processed_goalies = preprocess_player_stats(create_sample_dataframe('player_stats.csv'))
        
        # Prepare model data
        model_data = prepare_model_data(
            processed_schedule,
            processed_team_stats,
            processed_game_stats,
            recent_games
        )
        
        # Save processed data
        processed_data = {
            'schedule': processed_schedule,
            'game_stats': processed_game_stats,
            'team_stats': processed_team_stats,
            'skaters': processed_skaters,
            'goalies': processed_goalies,
            'model_data': model_data
        }
        
        for name, df in processed_data.items():
            save_dataframe(df, f"{name}.csv")
        
        return processed_data

if __name__ == "__main__":
    preprocess_data() 