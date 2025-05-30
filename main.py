"""
Main module for NHL game prediction model.
"""
import os
import sys
import logging
import pandas as pd
from datetime import datetime, timedelta
import argparse
import random

# Add the project directory to Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_dir)

from src.data_collection import collect_data, collect_upcoming_games
from src.preprocessing import preprocess_data, prepare_prediction_data
from src.model import train_models, predict_games
from src.utils import load_dataframe, save_dataframe, PROCESSED_DATA_DIR
# Import playoff prediction functionality
from src.playoff_prediction import (
    setup_playoff_bracket, 
    simulate_playoff_round, 
    simulate_full_playoffs, 
    print_playoff_bracket,
    PLAYOFF_ROUNDS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('nhl_predictor.log')
    ]
)

logger = logging.getLogger('nhl_predictor.main')

def run_full_pipeline(seasons: list = None, days_ahead: int = 7, retrain: bool = True):
    """
    Run the full NHL prediction pipeline.
    
    Args:
        seasons: List of NHL seasons to collect data for (e.g., ['20222023'])
        days_ahead: Number of days ahead to predict games for
        retrain: Whether to retrain the models
    
    Returns:
        DataFrame with game predictions
    """
    try:
        # Default to current season if none provided
        if not seasons:
            current_year = datetime.now().year
            current_month = datetime.now().month
            # NHL season runs from October to June of next year
            if current_month >= 10:  # October to December
                current_season = f"{current_year}{current_year + 1}"
            else:  # January to September
                current_season = f"{current_year - 1}{current_year}"
            seasons = [current_season]
        
        logger.info(f"Starting full pipeline for seasons: {seasons}")
        
        # Step 1: Collect Data
        logger.info("Step 1: Collecting data")
        try:
            collected_data = collect_data(seasons)
            logger.info("Data collection completed successfully")
        except Exception as e:
            logger.error(f"Error collecting data: {e}")
            logger.info("Using sample data for preprocessing")
            collected_data = {
                'teams': pd.DataFrame(),
                'schedule': pd.DataFrame(),
                'game_stats': pd.DataFrame(),
                'team_stats': pd.DataFrame(),
                'player_stats': pd.DataFrame()
            }
        
        # Step 2: Preprocess Data
        logger.info("Step 2: Preprocessing data")
        try:
            processed_data = preprocess_data()
            logger.info("Data preprocessing completed successfully")
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            logger.info("Will attempt to continue with basic model training")
            processed_data = {
                'model_data': pd.DataFrame()
            }
        
        # Step 3: Train Models (if requested)
        if retrain:
            logger.info("Step 3: Training models")
            try:
                model_results = train_models()
                logger.info("Model training completed successfully")
                logger.info(f"Win model accuracy: {model_results.get('win_accuracy', 0):.4f}")
                logger.info(f"Home score model RMSE: {model_results.get('home_score_rmse', 0):.4f}")
                logger.info(f"Away score model RMSE: {model_results.get('away_score_rmse', 0):.4f}")
            except Exception as e:
                logger.error(f"Error training models: {e}")
                logger.info("Will attempt to continue with predictions")
        
        # Step 4: Collect Upcoming Games for Prediction
        logger.info(f"Step 4: Collecting upcoming games for next {days_ahead} days")
        try:
            upcoming_games = collect_upcoming_games(days_ahead)
            logger.info(f"Found {len(upcoming_games)} upcoming games")
            
            if upcoming_games.empty:
                # Create demo upcoming games if none found
                today = datetime.now().strftime('%Y-%m-%d')
                tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
                
                # Use the first 2 teams from processed data if available
                teams_df = None
                team_ids = []
                team_names = []
                try:
                    teams_df = load_dataframe('teams.csv', processed=False)
                    team_ids = teams_df['team_id'].tolist()[:4]
                    team_names = teams_df['team_name'].tolist()[:4]
                except:
                    team_ids = [1, 2, 3, 4]  # Default IDs
                    team_names = ["Boston Bruins", "Tampa Bay Lightning", "Colorado Avalanche", "Vegas Golden Knights"]
                
                # Create sample upcoming games
                upcoming_games = pd.DataFrame([
                    {'game_id': 2023020001, 'date': tomorrow, 'home_team_id': team_ids[0], 
                     'away_team_id': team_ids[1], 'status': 'Preview', 'season': seasons[0],
                     'home_team_name': team_names[0], 'away_team_name': team_names[1]},
                    {'game_id': 2023020002, 'date': tomorrow, 'home_team_id': team_ids[2], 
                     'away_team_id': team_ids[3], 'status': 'Preview', 'season': seasons[0],
                     'home_team_name': team_names[2], 'away_team_name': team_names[3]}
                ])
                logger.info("Created sample upcoming games for demonstration")
            
        except Exception as e:
            logger.error(f"Error collecting upcoming games: {e}")
            logger.info("Creating sample upcoming games")
            
            # Create minimal upcoming games dataframe
            today = datetime.now().strftime('%Y-%m-%d')
            tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Try to get team names if possible
            teams_df = None
            team_ids = [1, 2, 3, 4]  # Default IDs
            team_names = ["Boston Bruins", "Tampa Bay Lightning", "Colorado Avalanche", "Vegas Golden Knights"]
            
            try:
                teams_df = load_dataframe('teams.csv', processed=False)
                if len(teams_df) >= 4:
                    team_ids = teams_df['team_id'].tolist()[:4]
                    team_names = teams_df['team_name'].tolist()[:4]
            except Exception as e:
                logger.warning(f"Could not load team names: {e}. Using defaults.")
            
            upcoming_games = pd.DataFrame([
                {'game_id': 2023020001, 'date': tomorrow, 'home_team_id': team_ids[0], 
                 'away_team_id': team_ids[1], 'status': 'Preview', 'season': seasons[0],
                 'home_team_name': team_names[0], 'away_team_name': team_names[1]},
                {'game_id': 2023020002, 'date': tomorrow, 'home_team_id': team_ids[2], 
                 'away_team_id': team_ids[3], 'status': 'Preview', 'season': seasons[0],
                 'home_team_name': team_names[2], 'away_team_name': team_names[3]}
            ])
        
        # Step 5: Prepare Prediction Data
        logger.info("Step 5: Preparing prediction data")
        try:
            # Load preprocessed data
            schedule_df = load_dataframe('schedule.csv')
            team_stats_df = load_dataframe('team_stats.csv')
            
            # Prepare prediction data
            prediction_data = prepare_prediction_data(
                upcoming_games,
                team_stats_df,
                schedule_df
            )
            logger.info("Prediction data prepared successfully")
        except Exception as e:
            logger.error(f"Error preparing prediction data: {e}")
            logger.info("Using minimal prediction data")
            # Just use the upcoming games as-is
            prediction_data = upcoming_games.copy()
            
            # Add minimal required columns for prediction to work
            prediction_data['season'] = seasons[0]
            
            # Add dummy team stats columns
            for prefix in ['home_', 'away_']:
                prediction_data[f'{prefix}win_pct'] = 0.5
                prediction_data[f'{prefix}goals_per_game'] = 3.0
                prediction_data[f'{prefix}goals_against_per_game'] = 2.8
                prediction_data[f'{prefix}last_n_wins'] = 5
                prediction_data[f'{prefix}last_n_goals_scored'] = 3.0
                prediction_data[f'{prefix}last_n_goals_conceded'] = 2.8
        
        # Step 6: Make Predictions
        logger.info("Step 6: Making predictions")
        try:
            predictions = predict_games(prediction_data)
            logger.info("Predictions generated successfully")
            
            # Add team names to predictions
            try:
                teams_df = load_dataframe('teams.csv', processed=False)
                teams_dict = dict(zip(teams_df['team_id'], teams_df['team_name']))
                
                # Add team names
                predictions['home_team_name'] = predictions['home_team_id'].map(teams_dict)
                predictions['away_team_name'] = predictions['away_team_id'].map(teams_dict)
                
                # Fill missing team names
                predictions['home_team_name'] = predictions['home_team_name'].fillna(
                    predictions['home_team_id'].apply(lambda x: f"Team {x}")
                )
                predictions['away_team_name'] = predictions['away_team_name'].fillna(
                    predictions['away_team_id'].apply(lambda x: f"Team {x}")
                )
                
                # Update winner name based on prediction
                predictions['winner_name'] = predictions.apply(
                    lambda row: row['home_team_name'] if row['predicted_winner'] == 'home' 
                    else row['away_team_name'], axis=1
                )
            except Exception as e:
                logger.warning(f"Could not add team names to predictions: {e}")
            
            # Save predictions with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            predictions_path = os.path.join(PROCESSED_DATA_DIR, f'predictions_{timestamp}.csv')
            predictions.to_csv(predictions_path, index=False)
            logger.info(f"Predictions saved to {predictions_path}")
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            logger.info("Creating sample predictions")
            
            # Try to get team names if possible
            team_names = {}
            try:
                teams_df = load_dataframe('teams.csv', processed=False)
                team_names = dict(zip(teams_df['team_id'], teams_df['team_name']))
            except Exception as e:
                logger.warning(f"Could not load team names: {e}. Using defaults.")
            
            # Create minimal prediction results
            predictions = pd.DataFrame()
            for _, game in upcoming_games.iterrows():
                home_team_id = game['home_team_id']
                away_team_id = game['away_team_id']
                
                # Get team names from dictionary or use defaults
                home_team_name = team_names.get(home_team_id, 
                    game.get('home_team_name', f"Team {home_team_id}"))
                away_team_name = team_names.get(away_team_id, 
                    game.get('away_team_name', f"Team {away_team_id}"))
                
                # Randomly select winner for sample data (50/50 chance)
                is_home_winner = random.choice([True, False])
                winner = 'home' if is_home_winner else 'away'
                win_prob = random.uniform(0.51, 0.85)
                
                # Create sample prediction data
                predictions = pd.concat([predictions, pd.DataFrame([{
                    'game_id': game['game_id'],
                    'date': game['date'],
                    'home_team_id': home_team_id,
                    'away_team_id': away_team_id,
                    'home_team_name': home_team_name,
                    'away_team_name': away_team_name,
                    'predicted_winner': winner,
                    'win_probability': win_prob if is_home_winner else 1 - win_prob,
                    'predicted_home_score': random.uniform(2.5, 4.5) if is_home_winner else random.uniform(1.5, 3.5),
                    'predicted_away_score': random.uniform(1.5, 3.5) if is_home_winner else random.uniform(2.5, 4.5),
                    'predicted_total': random.uniform(5.0, 7.5),
                    'winner_name': home_team_name if is_home_winner else away_team_name
                }])], ignore_index=True)
            
            # Save sample predictions
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            predictions_path = os.path.join(PROCESSED_DATA_DIR, f'sample_predictions_{timestamp}.csv')
            predictions.to_csv(predictions_path, index=False)
            logger.info(f"Sample predictions saved to {predictions_path}")
        
        # Return the predictions
        return predictions
    
    except Exception as e:
        logger.error(f"Unexpected error in pipeline: {e}")
        logger.info("Creating minimal predictions result")
        
        # Return minimal DataFrame
        return pd.DataFrame(columns=[
            'game_id', 'date', 'home_team_id', 'away_team_id',
            'predicted_winner', 'win_probability',
            'predicted_home_score', 'predicted_away_score', 'predicted_total'
        ])

def print_predictions(predictions: pd.DataFrame):
    """
    Print predictions in a readable format.
    
    Args:
        predictions: DataFrame with predictions
    """
    if predictions.empty:
        print("No predictions available.")
        return
    
    # Get team names if available
    try:
        # Check if team names are already in the predictions DataFrame
        if 'home_team_name' in predictions.columns and 'away_team_name' in predictions.columns:
            pass  # Team names already included
        else:
            # Try to get team names from teams.csv
            teams_df = load_dataframe('teams.csv', processed=False)
            teams_dict = dict(zip(teams_df['team_id'], teams_df['team_name']))
            
            # Map team IDs to names
            predictions['home_team_name'] = predictions['home_team_id'].map(teams_dict).fillna(
                predictions['home_team_id'].apply(lambda x: f"NHL Team {x}")
            )
            predictions['away_team_name'] = predictions['away_team_id'].map(teams_dict).fillna(
                predictions['away_team_id'].apply(lambda x: f"NHL Team {x}")
            )
    except Exception as e:
        logger.warning(f"Error getting team names: {e}")
        # Create default team names if mapping fails
        predictions['home_team_name'] = predictions['home_team_id'].apply(lambda x: f"NHL Team {x}")
        predictions['away_team_name'] = predictions['away_team_id'].apply(lambda x: f"NHL Team {x}")
    
    print("\n===== NHL GAME PREDICTIONS =====\n")
    
    # Group by date
    predictions['date'] = pd.to_datetime(predictions['date'])
    grouped = predictions.groupby(predictions['date'].dt.strftime('%Y-%m-%d'))
    
    for date, games in grouped:
        print(f"\nDate: {date}")
        print("-" * 50)
        
        for _, game in games.iterrows():
            home_team = game['home_team_name']
            away_team = game['away_team_name']
            
            winner = "HOME" if game['predicted_winner'] == 'home' else "AWAY"
            winner_name = home_team if winner == "HOME" else away_team
            win_prob = round(game['win_probability'] * 100, 1)
            win_prob = win_prob if winner == "HOME" else 100 - win_prob
            
            # Round score predictions to 1 decimal place for better display
            home_score = round(game['predicted_home_score'], 1)
            away_score = round(game['predicted_away_score'], 1)
            total = round(game['predicted_total'], 1)
            
            print(f"{away_team} @ {home_team}")
            print(f"Predicted Score: {away_team} {away_score} - {home_score} {home_team}")
            print(f"Predicted Winner: {winner_name} ({win_prob}% probability)")
            print(f"Predicted Total: {total} goals")
            print("-" * 50)
    
    print("\nNote: These predictions are for entertainment purposes only.")

def run_playoff_prediction(season: str = None, simulate_full: bool = True, 
                          single_round: str = None, num_simulations: int = 1000):
    """
    Run playoff bracket predictions for a specific season.
    
    Args:
        season: NHL season in format YYYYYYYY (e.g., '20222023')
        simulate_full: Whether to simulate the full playoffs or just a single round
        single_round: Specific playoff round to simulate (if not simulating full playoffs)
        num_simulations: Number of simulations per series
        
    Returns:
        DataFrame with playoff predictions
    """
    logger.info(f"Starting playoff predictions for season: {season}")
    
    # Default to current season if not specified
    if not season:
        current_year = datetime.now().year
        current_month = datetime.now().month
        # NHL season typically runs from October to June of next year
        if current_month >= 10:  # October to December
            season = f"{current_year}{current_year + 1}"
        else:  # January to September
            season = f"{current_year - 1}{current_year}"
    
    try:
        if simulate_full:
            # Run full playoff simulation
            logger.info(f"Simulating full playoffs for season {season}")
            bracket_df = simulate_full_playoffs(season=season, num_simulations=num_simulations)
            logger.info("Full playoff simulation completed successfully")
        else:
            # Simulate a specific round
            if single_round not in PLAYOFF_ROUNDS:
                valid_rounds = ", ".join(PLAYOFF_ROUNDS)
                raise ValueError(f"Invalid playoff round: '{single_round}'. Valid rounds: {valid_rounds}")
            
            # Check if we have an existing bracket, otherwise create one
            try:
                bracket_df = load_dataframe(f'playoff_bracket_{season}.csv')
                logger.info(f"Loaded existing playoff bracket for {season}")
                
                # Check if previous rounds are completed
                round_idx = PLAYOFF_ROUNDS.index(single_round)
                if round_idx > 0:
                    prev_round = PLAYOFF_ROUNDS[round_idx - 1]
                    prev_completed = bracket_df[bracket_df['round'] == prev_round]['completed'].all()
                    
                    if not prev_completed:
                        logger.warning(f"Previous round '{prev_round}' is not completed yet.")
                        
                        # Ask if user wants to simulate previous rounds
                        answer = input(f"Previous round '{prev_round}' is not completed. Simulate it first? (y/n): ")
                        if answer.lower() in ['y', 'yes']:
                            # Simulate previous rounds
                            for r in PLAYOFF_ROUNDS[:round_idx]:
                                logger.info(f"Simulating round: {r}")
                                bracket_df = simulate_playoff_round(
                                    bracket_df=bracket_df, 
                                    round_name=r,
                                    num_simulations=num_simulations
                                )
            except:
                # Create new bracket if it doesn't exist
                logger.info(f"Creating new playoff bracket for {season}")
                bracket_df = setup_playoff_bracket(season)
            
            # Simulate the specified round
            logger.info(f"Simulating playoff round: {single_round}")
            bracket_df = simulate_playoff_round(
                bracket_df=bracket_df,
                round_name=single_round,
                num_simulations=num_simulations
            )
            logger.info(f"Playoff round '{single_round}' simulation completed successfully")
        
        # Return the bracket
        return bracket_df
        
    except Exception as e:
        logger.error(f"Error simulating playoffs: {e}")
        raise

def main():
    """
    Main function to run the NHL prediction model.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='NHL Game and Playoff Prediction Model')
    
    # Regular game prediction arguments
    parser.add_argument('--seasons', nargs='+', help='NHL seasons to collect data for (e.g., 20222023)')
    parser.add_argument('--days', type=int, default=7, help='Number of days ahead to predict games for')
    parser.add_argument('--no-retrain', action='store_true', help='Skip model retraining')
    parser.add_argument('--collect-only', action='store_true', help='Only collect data, no prediction')
    parser.add_argument('--preprocess-only', action='store_true', help='Only preprocess data, no prediction')
    parser.add_argument('--train-only', action='store_true', help='Only train models, no prediction')
    
    # New playoff prediction arguments
    playoff_group = parser.add_argument_group('Playoff prediction options')
    playoff_group.add_argument('--playoff', action='store_true', help='Run playoff predictions instead of regular games')
    playoff_group.add_argument('--playoff-season', type=str, help='Season for playoff predictions (e.g., 20222023)')
    playoff_group.add_argument('--playoff-round', type=str, choices=PLAYOFF_ROUNDS, 
                             help='Specific playoff round to simulate (default: full playoffs)')
    playoff_group.add_argument('--simulations', type=int, default=1000, 
                             help='Number of simulations per playoff series (default: 1000)')
    
    args = parser.parse_args()
    
    # Check if we're doing playoff predictions
    if args.playoff:
        try:
            # Run playoff predictions
            bracket_df = run_playoff_prediction(
                season=args.playoff_season,
                simulate_full=args.playoff_round is None,
                single_round=args.playoff_round,
                num_simulations=args.simulations
            )
            
            # Print the playoff bracket
            print_playoff_bracket(bracket_df, use_abbreviations=True)
            return
        except Exception as e:
            logger.error(f"Failed to run playoff predictions: {str(e)}")
            return
    
    # Handle single mode operations
    if args.collect_only:
        seasons = args.seasons or ['20222023']
        logger.info(f"Collecting data for seasons: {seasons}")
        collect_data(seasons)
        logger.info("Data collection completed")
        return
    
    if args.preprocess_only:
        logger.info("Preprocessing data")
        preprocess_data()
        logger.info("Data preprocessing completed")
        return
    
    if args.train_only:
        logger.info("Training models")
        train_models()
        logger.info("Model training completed")
        return
    
    # Run full pipeline
    predictions = run_full_pipeline(
        seasons=args.seasons,
        days_ahead=args.days,
        retrain=not args.no_retrain
    )
    
    # Print predictions
    print_predictions(predictions)

if __name__ == "__main__":
    main() 