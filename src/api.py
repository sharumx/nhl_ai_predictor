"""
API module for NHL game prediction model.
"""
import os
import logging
import pandas as pd
from flask import Flask, jsonify, request
from datetime import datetime

from src.utils import load_dataframe, PROCESSED_DATA_DIR
from src.data_collection import collect_upcoming_games
from src.preprocessing import prepare_prediction_data
from src.model import predict_games

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('nhl_api.log')
    ]
)
logger = logging.getLogger('nhl_predictor.api')

# Global variables to cache results
CACHE_TIMEOUT_SECONDS = 3600  # 1 hour
last_prediction_time = None
cached_predictions = None

@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint."""
    return jsonify({
        'status': 'ok',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    """Get predictions for upcoming games."""
    global last_prediction_time, cached_predictions
    
    # Check if we need to refresh predictions
    current_time = datetime.now()
    if (last_prediction_time is None or 
        cached_predictions is None or 
        (current_time - last_prediction_time).total_seconds() > CACHE_TIMEOUT_SECONDS):
        
        # Get days parameter
        days = request.args.get('days', default=7, type=int)
        
        try:
            # Get fresh predictions
            logger.info(f"Generating new predictions for next {days} days")
            
            # Try to collect upcoming games
            try:
                upcoming_games = collect_upcoming_games(days)
                
                if upcoming_games.empty:
                    return jsonify({
                        'status': 'error',
                        'message': 'No upcoming games found',
                        'timestamp': current_time.isoformat()
                    }), 404
                
            except Exception as e:
                logger.error(f"Error collecting upcoming games: {e}")
                return jsonify({
                    'status': 'error',
                    'message': f'Failed to collect upcoming games: {str(e)}',
                    'timestamp': current_time.isoformat()
                }), 500
            
            # Load team stats
            try:
                schedule_df = load_dataframe('schedule.csv')
                team_stats_df = load_dataframe('team_stats.csv')
            except Exception as e:
                logger.error(f"Error loading preprocessed data: {e}")
                return jsonify({
                    'status': 'error',
                    'message': f'Failed to load preprocessed data: {str(e)}',
                    'timestamp': current_time.isoformat()
                }), 500
            
            # Prepare prediction data
            try:
                prediction_data = prepare_prediction_data(
                    upcoming_games,
                    team_stats_df,
                    schedule_df
                )
            except Exception as e:
                logger.error(f"Error preparing prediction data: {e}")
                return jsonify({
                    'status': 'error',
                    'message': f'Failed to prepare prediction data: {str(e)}',
                    'timestamp': current_time.isoformat()
                }), 500
            
            # Make predictions
            try:
                predictions = predict_games(prediction_data)
                
                # Cache the results
                last_prediction_time = current_time
                cached_predictions = predictions.to_dict(orient='records')
                
                # Save predictions
                timestamp = current_time.strftime('%Y%m%d_%H%M%S')
                predictions_path = os.path.join(PROCESSED_DATA_DIR, f'api_predictions_{timestamp}.csv')
                predictions.to_csv(predictions_path, index=False)
                logger.info(f"Predictions saved to {predictions_path}")
                
            except Exception as e:
                logger.error(f"Error making predictions: {e}")
                return jsonify({
                    'status': 'error',
                    'message': f'Failed to make predictions: {str(e)}',
                    'timestamp': current_time.isoformat()
                }), 500
        except Exception as e:
            logger.error(f"Unexpected error in prediction process: {e}")
            return jsonify({
                'status': 'error',
                'message': f'Unexpected error: {str(e)}',
                'timestamp': current_time.isoformat()
            }), 500
    else:
        logger.info("Using cached predictions")
    
    # Get team information if available
    try:
        teams_df = load_dataframe('teams.csv', processed=False)
        teams_dict = {
            int(team_id): {
                'name': team_name,
                'abbreviation': team_abbrev
            } for team_id, team_name, team_abbrev in zip(
                teams_df['team_id'], 
                teams_df['team_name'], 
                teams_df['team_abbrev']
            )
        }
    except Exception as e:
        logger.warning(f"Could not load team information: {e}")
        teams_dict = {}
    
    # Enhance predictions with team names
    enhanced_predictions = []
    for pred in cached_predictions:
        home_team_id = pred.get('home_team_id')
        away_team_id = pred.get('away_team_id')
        
        enhanced_pred = pred.copy()
        enhanced_pred['home_team'] = teams_dict.get(home_team_id, {}).get('name', f'Team {home_team_id}')
        enhanced_pred['away_team'] = teams_dict.get(away_team_id, {}).get('name', f'Team {away_team_id}')
        enhanced_pred['home_team_abbrev'] = teams_dict.get(home_team_id, {}).get('abbreviation', f'T{home_team_id}')
        enhanced_pred['away_team_abbrev'] = teams_dict.get(away_team_id, {}).get('abbreviation', f'T{away_team_id}')
        
        enhanced_predictions.append(enhanced_pred)
    
    # Return response
    return jsonify({
        'status': 'ok',
        'timestamp': current_time.isoformat(),
        'cache_time': last_prediction_time.isoformat() if last_prediction_time else None,
        'predictions': enhanced_predictions
    })

@app.route('/api/teams', methods=['GET'])
def get_teams():
    """Get NHL teams."""
    try:
        teams_df = load_dataframe('teams.csv', processed=False)
        teams = teams_df.to_dict(orient='records')
        
        return jsonify({
            'status': 'ok',
            'timestamp': datetime.now().isoformat(),
            'teams': teams
        })
    except Exception as e:
        logger.error(f"Error retrieving teams: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to retrieve teams: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/predictions/latest', methods=['GET'])
def get_latest_predictions():
    """Get the most recent predictions file."""
    try:
        # Find the most recent predictions file
        prediction_files = [f for f in os.listdir(PROCESSED_DATA_DIR) 
                          if f.startswith('predictions_') and f.endswith('.csv')]
        
        if not prediction_files:
            return jsonify({
                'status': 'error',
                'message': 'No prediction files found',
                'timestamp': datetime.now().isoformat()
            }), 404
        
        # Sort by timestamp in filename
        latest_file = sorted(prediction_files, reverse=True)[0]
        latest_path = os.path.join(PROCESSED_DATA_DIR, latest_file)
        
        # Load the file
        predictions_df = pd.read_csv(latest_path)
        predictions = predictions_df.to_dict(orient='records')
        
        # Get team information if available
        try:
            teams_df = load_dataframe('teams.csv', processed=False)
            teams_dict = {
                int(team_id): {
                    'name': team_name,
                    'abbreviation': team_abbrev
                } for team_id, team_name, team_abbrev in zip(
                    teams_df['team_id'], 
                    teams_df['team_name'], 
                    teams_df['team_abbrev']
                )
            }
        except Exception as e:
            logger.warning(f"Could not load team information: {e}")
            teams_dict = {}
        
        # Enhance predictions with team names
        enhanced_predictions = []
        for pred in predictions:
            home_team_id = pred.get('home_team_id')
            away_team_id = pred.get('away_team_id')
            
            enhanced_pred = pred.copy()
            enhanced_pred['home_team'] = teams_dict.get(home_team_id, {}).get('name', f'Team {home_team_id}')
            enhanced_pred['away_team'] = teams_dict.get(away_team_id, {}).get('name', f'Team {away_team_id}')
            enhanced_pred['home_team_abbrev'] = teams_dict.get(home_team_id, {}).get('abbreviation', f'T{home_team_id}')
            enhanced_pred['away_team_abbrev'] = teams_dict.get(away_team_id, {}).get('abbreviation', f'T{away_team_id}')
            
            enhanced_predictions.append(enhanced_pred)
        
        # Create timestamp from filename
        file_timestamp = latest_file.replace('predictions_', '').replace('.csv', '')
        
        return jsonify({
            'status': 'ok',
            'file': latest_file,
            'file_timestamp': file_timestamp,
            'timestamp': datetime.now().isoformat(),
            'predictions': enhanced_predictions
        })
    except Exception as e:
        logger.error(f"Error retrieving latest predictions: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to retrieve latest predictions: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    # Create data directories if they don't exist
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5001) 