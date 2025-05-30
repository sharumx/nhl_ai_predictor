"""
Playoff prediction functionality for NHL prediction model.
This module simulates playoff series and full playoff brackets.
"""
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple, Any

from src.model import load_models, predict_single_game
from src.utils import load_dataframe, save_dataframe, PROCESSED_DATA_DIR

# Configure logging
logger = logging.getLogger('nhl_predictor.playoff')

# Constants
PLAYOFF_ROUNDS = ["First Round", "Second Round", "Conference Finals", "Stanley Cup Final"]
PLAYOFF_FORMAT = {
    "First Round": 8,       # 8 series
    "Second Round": 4,      # 4 series
    "Conference Finals": 2, # 2 conference finals
    "Stanley Cup Final": 1  # 1 final series
}
GAMES_TO_WIN = 4  # Best of 7 series
HOME_ADVANTAGE = 0.05  # Additional win probability for home games

def setup_playoff_bracket(season: str = None) -> pd.DataFrame:
    """
    Set up the playoff bracket for a given season.
    If no season is specified, use the most recent season available.
    
    Args:
        season: Season in format YYYYYYYY (e.g., '20222023')
        
    Returns:
        DataFrame representing the playoff bracket
    """
    # If no season specified, use current season
    if not season:
        current_year = datetime.now().year
        current_month = datetime.now().month
        # NHL season typically runs from October to June of next year
        if current_month >= 10:  # October to December
            season = f"{current_year}{current_year + 1}"
        else:  # January to September
            season = f"{current_year - 1}{current_year}"
    
    logger.info(f"Setting up playoff bracket for season {season}")
    
    try:
        # Load teams and standings data
        teams_df = load_dataframe('teams.csv', processed=False)
        
        # Try to load team stats for the season to get standings
        try:
            team_stats_df = load_dataframe('team_stats.csv', processed=True)
            # Ensure team IDs are integers to match teams.csv
            team_stats_df['team_id'] = team_stats_df['team_id'].astype(int)
            
            # Check if the season column exists and has meaningful data
            if 'season' in team_stats_df.columns:
                # Get unique seasons
                available_seasons = team_stats_df['season'].unique()
                logger.info(f"Available seasons in team stats: {available_seasons}")
                logger.info(f"Season types in team stats: {[type(s) for s in available_seasons]}")
                logger.info(f"Current season we're looking for: {season} (type: {type(season)})")
                
                # Try different approaches to match the season
                season_matched = False
                
                # Try direct comparison
                if season in available_seasons:
                    logger.info(f"Direct season match found for {season}")
                    team_stats_df = team_stats_df[team_stats_df['season'] == season]
                    season_matched = True
                    
                # Try string comparison if needed
                elif str(season) in [str(s) for s in available_seasons]:
                    logger.info(f"String-based season match found for {season}")
                    team_stats_df = team_stats_df[team_stats_df['season'].astype(str) == str(season)]
                    season_matched = True
                    
                # Try integer comparison if needed
                elif int(season) in [int(s) for s in available_seasons if str(s).isdigit()]:
                    logger.info(f"Integer-based season match found for {season}")
                    team_stats_df = team_stats_df[team_stats_df['season'].astype(int) == int(season)]
                    season_matched = True
                
                # If no match found, use the most recent season
                if not season_matched:
                    logger.warning(f"Season {season} not found in team stats. Using most recent available season.")
                    if len(available_seasons) > 0:
                        # Try to convert to integers for comparison
                        try:
                            int_seasons = [int(s) for s in available_seasons if str(s).isdigit()]
                            if int_seasons:
                                latest_season = max(int_seasons)
                                team_stats_df = team_stats_df[team_stats_df['season'] == latest_season]
                                logger.info(f"Using data from season {latest_season}, found {len(team_stats_df)} teams")
                        except Exception as e:
                            logger.warning(f"Error finding latest season: {e}. Using all seasons.")
                
                logger.info(f"After season filtering, found {len(team_stats_df)} teams")
            
            # Sort by points to get standings
            team_stats_df = team_stats_df.sort_values(by='pts', ascending=False)
            
            # Make sure all team IDs in team_stats_df exist in teams_df
            valid_team_ids = set(teams_df['team_id'].astype(int))
            original_count = len(team_stats_df)
            team_stats_df = team_stats_df[team_stats_df['team_id'].isin(valid_team_ids)]
            if len(team_stats_df) < original_count:
                logger.warning(f"Removed {original_count - len(team_stats_df)} teams with IDs not found in teams.csv")
            
            # Add conference and division if not present in team_stats_df
            if 'conference' not in team_stats_df.columns or 'division' not in team_stats_df.columns:
                logger.info("Adding conference and division data from teams.csv")
                # Create temp dict for mapping
                conf_dict = dict(zip(teams_df['team_id'], teams_df['conference']))
                div_dict = dict(zip(teams_df['team_id'], teams_df['division']))
                
                # Add columns
                if 'conference' not in team_stats_df.columns:
                    team_stats_df['conference'] = team_stats_df['team_id'].map(conf_dict)
                if 'division' not in team_stats_df.columns:
                    team_stats_df['division'] = team_stats_df['team_id'].map(div_dict)
            
        except Exception as e:
            logger.error(f"Error loading team stats: {e}")
            logger.info("Creating team stats using the team information from teams.csv")
            # Create sample team stats using the actual teams from teams.csv
            team_stats_df = teams_df[['team_id']].copy()
            team_stats_df['season'] = season
            team_stats_df['pts'] = [random.randint(70, 120) for _ in range(len(team_stats_df))]
            team_stats_df['conference'] = teams_df['conference']
            team_stats_df['division'] = teams_df['division']
            team_stats_df = team_stats_df.sort_values(by='pts', ascending=False)
        
        # Count how many valid teams we have for playoffs
        logger.info(f"Found {len(team_stats_df)} teams for playoff bracket")
        
        # If we don't have enough teams, add missing teams from teams.csv
        if len(team_stats_df) < 16:
            logger.warning(f"Not enough teams for a full playoff bracket. Adding missing teams from teams.csv.")
            # Get team IDs already in the stats dataframe
            existing_team_ids = set(team_stats_df['team_id'])
            # Find teams in teams.csv that aren't in the stats dataframe
            missing_team_ids = set(teams_df['team_id']) - existing_team_ids
            
            # Add missing teams with lower points
            missing_teams = []
            for team_id in missing_team_ids:
                team_row = teams_df[teams_df['team_id'] == team_id].iloc[0]
                missing_teams.append({
                    'team_id': team_id,
                    'season': season,
                    'pts': random.randint(60, 85),  # Lower points for added teams
                    'conference': team_row['conference'],
                    'division': team_row['division'],
                    'games_played': 82,
                    'wins': random.randint(25, 38),
                    'losses': random.randint(30, 45),
                    'ot_losses': random.randint(5, 15),
                    'goals_per_game': round(random.uniform(2.5, 3.0), 1),
                    'goals_against_per_game': round(random.uniform(3.0, 3.5), 1),
                    'powerplay_pct': round(random.uniform(18.0, 22.0), 1),
                    'penalty_kill_pct': round(random.uniform(75.0, 82.0), 1),
                    'shots_per_game': round(random.uniform(28.0, 32.0), 1),
                    'shots_allowed': round(random.uniform(29.0, 33.0), 1)
                })
            
            if missing_teams:
                missing_df = pd.DataFrame(missing_teams)
                team_stats_df = pd.concat([team_stats_df, missing_df], ignore_index=True)
                logger.info(f"Added {len(missing_teams)} missing teams from teams.csv")
        
        # Sort by points and ensure we have exactly 16 teams
        team_stats_df = team_stats_df.sort_values(by='pts', ascending=False)
        if len(team_stats_df) > 16:
            # If we have more than 16 teams, keep only the top 16
            logger.warning(f"More than 16 teams available. Using only the top 16 by points.")
            team_stats_df = team_stats_df.head(16)
        elif len(team_stats_df) < 16:
            # This shouldn't happen now, but just in case
            logger.error(f"Still only have {len(team_stats_df)} teams after adding missing teams. This is unexpected.")
        else:
            logger.info("Exactly 16 teams available for playoffs, perfect!")
        
        # Create playoff bracket
        # Structure: Round, Series#, HomeTeamID, AwayTeamID, HomeWins, AwayWins, Winner, Completed
        playoff_bracket = []
        
        # Make sure conferences are balanced - we need 8 teams in each conference
        eastern_teams = team_stats_df[team_stats_df['conference'].str.contains('East', case=False)]
        western_teams = team_stats_df[team_stats_df['conference'].str.contains('West', case=False)]
        
        # Log conference balance
        logger.info(f"Conference balance before adjustment: Eastern={len(eastern_teams)}, Western={len(western_teams)}")
        
        # If conferences are unbalanced, adjust
        if len(eastern_teams) != 8 or len(western_teams) != 8:
            logger.warning(f"Conferences are unbalanced. Adjusting to ensure 8 teams in each conference.")
            if len(eastern_teams) > 8:
                logger.info(f"Too many Eastern teams ({len(eastern_teams)}). Keeping top 8 by points.")
                eastern_teams = eastern_teams.sort_values(by='pts', ascending=False).head(8)
            if len(western_teams) > 8:
                logger.info(f"Too many Western teams ({len(western_teams)}). Keeping top 8 by points.")
                western_teams = western_teams.sort_values(by='pts', ascending=False).head(8)
                
            # If we don't have enough teams in a conference, move teams from the other conference
            while len(eastern_teams) < 8 and len(western_teams) > 8:
                logger.info("Moving a Western team to Eastern conference to balance.")
                # Move the lowest-ranked Western team to Eastern
                team_to_move = western_teams.sort_values(by='pts').iloc[0]
                eastern_teams = pd.concat([eastern_teams, pd.DataFrame([team_to_move])], ignore_index=True)
                western_teams = western_teams[western_teams['team_id'] != team_to_move['team_id']]
                
            while len(western_teams) < 8 and len(eastern_teams) > 8:
                logger.info("Moving an Eastern team to Western conference to balance.")
                # Move the lowest-ranked Eastern team to Western
                team_to_move = eastern_teams.sort_values(by='pts').iloc[0]
                western_teams = pd.concat([western_teams, pd.DataFrame([team_to_move])], ignore_index=True)
                eastern_teams = eastern_teams[eastern_teams['team_id'] != team_to_move['team_id']]
                
            # Log final conference balance
            logger.info(f"Conference balance after adjustment: Eastern={len(eastern_teams)}, Western={len(western_teams)}")
        else:
            logger.info("Conference balance is perfect: 8 Eastern and 8 Western teams.")
        
        # First round matchups (1v8, 2v7, 3v6, 4v5 in each conference)
        for conf_idx, conf_teams in enumerate([eastern_teams, western_teams]):
            conf_prefix = "E" if conf_idx == 0 else "W"
            conf_teams = conf_teams.sort_values(by='pts', ascending=False).reset_index(drop=True)
            
            for i in range(4):
                home_team = int(conf_teams.iloc[i]['team_id'])
                away_team = int(conf_teams.iloc[7-i]['team_id'])
                series_id = f"{conf_prefix}{i+1}"
                
                playoff_bracket.append({
                    'round': 'First Round',
                    'series_id': series_id,
                    'home_team_id': home_team,
                    'away_team_id': away_team,
                    'home_wins': 0,
                    'away_wins': 0,
                    'winner_id': None,
                    'completed': False,
                    'season': season
                })
        
        # Create empty slots for subsequent rounds
        for round_name, num_series in list(PLAYOFF_FORMAT.items())[1:]:  # Skip first round
            for i in range(num_series):
                if round_name == "Second Round":
                    conf_prefix = "E" if i < 2 else "W"
                    series_id = f"{conf_prefix}{i%2+1}"
                elif round_name == "Conference Finals":
                    series_id = "E" if i == 0 else "W"
                else:  # Stanley Cup Final
                    series_id = "F"
                
                playoff_bracket.append({
                    'round': round_name,
                    'series_id': series_id,
                    'home_team_id': None,
                    'away_team_id': None,
                    'home_wins': 0,
                    'away_wins': 0,
                    'winner_id': None,
                    'completed': False,
                    'season': season
                })
        
        # Convert to DataFrame
        bracket_df = pd.DataFrame(playoff_bracket)
        
        # Add team names and abbreviations from teams_df
        teams_dict = dict(zip(teams_df['team_id'], teams_df['team_name']))
        abbrev_dict = dict(zip(teams_df['team_id'], teams_df['team_abbrev']))
        
        bracket_df['home_team_name'] = bracket_df['home_team_id'].map(teams_dict)
        bracket_df['away_team_name'] = bracket_df['away_team_id'].map(teams_dict)
        bracket_df['home_team_abbrev'] = bracket_df['home_team_id'].map(abbrev_dict)
        bracket_df['away_team_abbrev'] = bracket_df['away_team_id'].map(abbrev_dict)
        
        # Save bracket
        save_dataframe(bracket_df, f'playoff_bracket_{season}.csv')
        logger.info(f"Playoff bracket for season {season} created successfully")
        
        return bracket_df
        
    except Exception as e:
        logger.error(f"Error setting up playoff bracket: {e}")
        raise

def simulate_playoff_series(home_team_id: int, away_team_id: int, 
                           team_stats_df: pd.DataFrame, model_data: Dict, 
                           num_simulations: int = 100) -> Dict:
    """
    Simulate a playoff series between two teams.
    
    Args:
        home_team_id: ID of the team with home-ice advantage
        away_team_id: ID of the away team
        team_stats_df: DataFrame with team statistics
        model_data: Dict containing the prediction models
        num_simulations: Number of series to simulate
        
    Returns:
        Dict with series simulation results
    """
    logger.info(f"Simulating playoff series: Team {home_team_id} vs Team {away_team_id}")
    
    home_team_wins = 0
    away_team_wins = 0
    series_lengths = []
    
    for sim in range(num_simulations):
        series_home_wins = 0
        series_away_wins = 0
        games_played = 0
        
        # In NHL playoffs, home-ice advantage follows this pattern for a 7-game series:
        # Game 1: Home, Game 2: Home, Game 3: Away, Game 4: Away,
        # Game 5: Home, Game 6: Away, Game 7: Home
        home_pattern = [True, True, False, False, True, False, True]
        
        while series_home_wins < GAMES_TO_WIN and series_away_wins < GAMES_TO_WIN:
            # Determine if current game is at home or away
            is_home_game = home_pattern[games_played % len(home_pattern)]
            
            # Create game prediction data
            game_data = {
                'home_team_id': home_team_id if is_home_game else away_team_id,
                'away_team_id': away_team_id if is_home_game else home_team_id
            }
            
            # Use actual model to predict if available
            try:
                prediction = predict_single_game(game_data, model_data)
                win_prob = prediction['win_probability']
                
                # Apply home ice advantage boost for playoff games
                if is_home_game:
                    win_prob = min(win_prob + HOME_ADVANTAGE, 0.95)  # Cap at 95%
                else:
                    win_prob = max(win_prob - HOME_ADVANTAGE, 0.05)  # Floor at 5%
                    
                home_team_game_win = random.random() < win_prob
            except Exception as e:
                logger.warning(f"Error using model for prediction: {e}. Using random simulation.")
                # Fallback to random simulation with slight home advantage
                home_team_game_win = random.random() < (0.55 if is_home_game else 0.45)
            
            # Update series score
            if (is_home_game and home_team_game_win) or (not is_home_game and not home_team_game_win):
                series_home_wins += 1
            else:
                series_away_wins += 1
            
            games_played += 1
        
        # Record series outcome
        if series_home_wins > series_away_wins:
            home_team_wins += 1
        else:
            away_team_wins += 1
        
        series_lengths.append(games_played)
    
    # Calculate series stats
    home_win_pct = home_team_wins / num_simulations
    avg_series_length = sum(series_lengths) / len(series_lengths)
    
    # Common series outcomes
    series_outcomes = {
        "4-0": 0, "4-1": 0, "4-2": 0, "4-3": 0,
        "0-4": 0, "1-4": 0, "2-4": 0, "3-4": 0
    }
    
    # Return results
    return {
        'home_team_id': home_team_id,
        'away_team_id': away_team_id,
        'home_win_probability': home_win_pct,
        'away_win_probability': 1 - home_win_pct,
        'predicted_winner_id': home_team_id if home_win_pct > 0.5 else away_team_id,
        'avg_series_length': avg_series_length,
        'num_simulations': num_simulations
    }

def simulate_playoff_round(bracket_df: pd.DataFrame, round_name: str, 
                          team_stats_df: pd.DataFrame = None, 
                          num_simulations: int = 100) -> pd.DataFrame:
    """
    Simulate a full playoff round.
    
    Args:
        bracket_df: DataFrame with the playoff bracket
        round_name: Name of the round to simulate
        team_stats_df: DataFrame with team statistics
        num_simulations: Number of simulations per series
        
    Returns:
        Updated bracket DataFrame
    """
    logger.info(f"Simulating playoff round: {round_name}")
    
    # Load models for prediction
    try:
        model_data = load_models()
        logger.info("Loaded prediction models successfully")
    except Exception as e:
        logger.warning(f"Error loading models: {e}. Using random simulation.")
        model_data = None
    
    # Load team stats if not provided
    if team_stats_df is None:
        try:
            team_stats_df = load_dataframe('team_stats.csv', processed=True)
            # Ensure team IDs are integers
            team_stats_df['team_id'] = team_stats_df['team_id'].astype(int)
        except Exception as e:
            logger.warning(f"Error loading team stats: {e}")
            team_stats_df = pd.DataFrame()
    
    # Load teams data for name lookups
    teams_df = load_dataframe('teams.csv', processed=False)
    teams_dict = dict(zip(teams_df['team_id'], teams_df['team_name']))
    abbrev_dict = dict(zip(teams_df['team_id'], teams_df['team_abbrev']))
    
    # Ensure team_id columns are integers in bracket_df
    id_columns = ['home_team_id', 'away_team_id', 'winner_id']
    for col in id_columns:
        if col in bracket_df.columns:
            # Convert to int only where not null
            mask = ~bracket_df[col].isna()
            if mask.any():
                bracket_df.loc[mask, col] = bracket_df.loc[mask, col].astype(int)
    
    # Filter to current round and incomplete series
    round_mask = (bracket_df['round'] == round_name) & (~bracket_df['completed'])
    current_series = bracket_df[round_mask].copy()
    
    # Check if we need to populate teams from previous round
    if round_name != "First Round":
        # Find previous round
        round_idx = PLAYOFF_ROUNDS.index(round_name)
        prev_round = PLAYOFF_ROUNDS[round_idx - 1]
        
        # Get completed series from previous round
        prev_round_mask = (bracket_df['round'] == prev_round) & (bracket_df['completed'])
        prev_series = bracket_df[prev_round_mask].copy()
        
        # Populate teams for current round based on previous winners
        if not prev_series.empty:
            # Match series winners to next round
            for idx, series in current_series.iterrows():
                series_id = series['series_id']
                
                # Find contributing series from previous round
                if round_name == "Second Round":
                    # For Second Round: 
                    # E1 gets winners of E1 and E4
                    # E2 gets winners of E2 and E3
                    # W1 gets winners of W1 and W4
                    # W2 gets winners of W2 and W3
                    if series_id in ["E1", "W1"]:
                        contrib_ids = [f"{series_id[0]}1", f"{series_id[0]}4"]
                    else:  # E2 or W2
                        contrib_ids = [f"{series_id[0]}2", f"{series_id[0]}3"]
                elif round_name == "Conference Finals":
                    # For Conference Finals:
                    # E gets winners of E1 and E2
                    # W gets winners of W1 and W2
                    contrib_ids = [f"{series_id}1", f"{series_id}2"]
                else:  # Stanley Cup Final
                    # For Final:
                    # F gets winners of E and W
                    contrib_ids = ["E", "W"]
                
                # Get winners from contributing series
                contributors = prev_series[prev_series['series_id'].isin(contrib_ids)].copy()
                
                if len(contributors) == 2:  # We have both teams
                    # Ensure winner_id is integer
                    contributors['winner_id'] = contributors['winner_id'].astype(int)
                    
                    # Sort contributors by points if available
                    if not team_stats_df.empty:
                        # Try to get team stats for sorting
                        team_ids = contributors['winner_id'].tolist()
                        team_points = []
                        
                        for team_id in team_ids:
                            try:
                                team_pts = team_stats_df[team_stats_df['team_id'] == team_id]['pts'].values[0]
                                team_points.append(team_pts)
                            except:
                                # If no stats, just use random points
                                team_points.append(random.randint(80, 110))
                        
                        # Sort teams by points
                        sorted_teams = [x for _, x in sorted(zip(team_points, team_ids), reverse=True)]
                        home_team, away_team = sorted_teams
                    else:
                        # No team stats, just use first as home
                        home_team = int(contributors.iloc[0]['winner_id'])
                        away_team = int(contributors.iloc[1]['winner_id'])
                    
                    # Update bracket with teams
                    bracket_df.loc[idx, 'home_team_id'] = home_team
                    bracket_df.loc[idx, 'away_team_id'] = away_team
                    
                    # Add team names
                    bracket_df.loc[idx, 'home_team_name'] = teams_dict.get(home_team, f"Team {home_team}")
                    bracket_df.loc[idx, 'away_team_name'] = teams_dict.get(away_team, f"Team {away_team}")
                    bracket_df.loc[idx, 'home_team_abbrev'] = abbrev_dict.get(home_team, f"T{home_team}")
                    bracket_df.loc[idx, 'away_team_abbrev'] = abbrev_dict.get(away_team, f"T{away_team}")
    
    # Now simulate each series in the round
    for idx, series in bracket_df[round_mask].iterrows():
        home_team_id = series['home_team_id']
        away_team_id = series['away_team_id']
        
        # Skip if teams aren't populated yet
        if pd.isna(home_team_id) or pd.isna(away_team_id):
            continue
        
        # Ensure team IDs are integers
        home_team_id = int(home_team_id)
        away_team_id = int(away_team_id)
        
        # Simulate series
        result = simulate_playoff_series(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            team_stats_df=team_stats_df,
            model_data=model_data,
            num_simulations=num_simulations
        )
        
        # Determine winner based on higher win probability
        winner_id = int(result['predicted_winner_id'])
        
        # Update bracket with results
        bracket_df.loc[idx, 'winner_id'] = winner_id
        bracket_df.loc[idx, 'completed'] = True
        bracket_df.loc[idx, 'winner_name'] = teams_dict.get(winner_id, f"Team {winner_id}")
        bracket_df.loc[idx, 'winner_abbrev'] = abbrev_dict.get(winner_id, f"T{winner_id}")
        
        # Set a simulated final series score (for display purposes)
        # 4-0, 4-1, 4-2, or 4-3
        possible_scores = [(4,0), (4,1), (4,2), (4,3)]
        weights = [0.15, 0.25, 0.35, 0.25]  # More likely to be 4-2
        home_wins, away_wins = random.choices(possible_scores, weights)[0]
        
        # If away team won, flip the score
        if winner_id == away_team_id:
            home_wins, away_wins = away_wins, home_wins
            
        bracket_df.loc[idx, 'home_wins'] = home_wins
        bracket_df.loc[idx, 'away_wins'] = away_wins
    
    # Save updated bracket
    season = bracket_df['season'].iloc[0]
    save_dataframe(bracket_df, f'playoff_bracket_{season}.csv')
    
    return bracket_df

def simulate_full_playoffs(season: str = None, num_simulations: int = 100) -> pd.DataFrame:
    """
    Simulate entire NHL playoffs from first round through Stanley Cup Final.
    
    Args:
        season: Season in format YYYYYYYY (e.g., '20222023')
        num_simulations: Number of simulations per series
        
    Returns:
        DataFrame with complete playoff results
    """
    logger.info(f"Simulating full playoffs for season {season}")
    
    # Set up initial bracket
    bracket_df = setup_playoff_bracket(season)
    
    # Load team stats
    try:
        team_stats_df = load_dataframe('team_stats.csv', processed=True)
        # Ensure team IDs are integers
        team_stats_df['team_id'] = team_stats_df['team_id'].astype(int)
        
        if season and 'season' in team_stats_df.columns:
            team_stats_df = team_stats_df[team_stats_df['season'] == season]
    except Exception as e:
        logger.warning(f"Error loading team stats: {e}")
        team_stats_df = pd.DataFrame()
    
    # Load teams data for name lookups
    teams_df = load_dataframe('teams.csv', processed=False)
    teams_dict = dict(zip(teams_df['team_id'], teams_df['team_name']))
    
    # Simulate each round sequentially
    for round_name in PLAYOFF_ROUNDS:
        logger.info(f"Simulating {round_name}")
        bracket_df = simulate_playoff_round(
            bracket_df=bracket_df,
            round_name=round_name,
            team_stats_df=team_stats_df,
            num_simulations=num_simulations
        )
    
    # Save final results
    results_path = os.path.join(PROCESSED_DATA_DIR, f"playoff_results_{season}_{datetime.now().strftime('%Y%m%d')}.csv")
    bracket_df.to_csv(results_path, index=False)
    logger.info(f"Playoff simulation completed. Results saved to {results_path}")
    
    # Get the Champion (winner of Stanley Cup Final)
    champion_row = bracket_df[bracket_df['round'] == "Stanley Cup Final"].iloc[0]
    
    # Ensure we have the champion's name
    champion_id = int(champion_row['winner_id']) if not pd.isna(champion_row['winner_id']) else None
    if champion_id and 'winner_name' not in champion_row:
        champion_name = teams_dict.get(champion_id, f"Team {champion_id}")
    else:
        champion_name = champion_row.get('winner_name', "Unknown")
    
    logger.info(f"Predicted Stanley Cup Champion: {champion_name} (ID: {champion_id})")
    
    return bracket_df

def print_playoff_bracket(bracket_df: pd.DataFrame, use_abbreviations: bool = False) -> None:
    """
    Print the playoff bracket in a readable format.
    
    Args:
        bracket_df: DataFrame with playoff bracket
        use_abbreviations: Whether to use team abbreviations instead of full names
    """
    if bracket_df.empty:
        print("No playoff bracket available.")
        return
    
    # Make sure we have team names and abbreviations for all teams
    teams_df = load_dataframe('teams.csv', processed=False)
    teams_dict = dict(zip(teams_df['team_id'], teams_df['team_name']))
    abbrev_dict = dict(zip(teams_df['team_id'], teams_df['team_abbrev']))
    
    # Fill in any missing team names and abbreviations
    for col in ['home_team_id', 'away_team_id', 'winner_id']:
        if col in bracket_df.columns:
            name_col = col.replace('_id', '_name')
            abbrev_col = col.replace('_id', '_abbrev')
            
            # Create or update columns where needed
            for idx, row in bracket_df.iterrows():
                team_id = row.get(col)
                
                # Skip if no team ID
                if pd.isna(team_id):
                    continue
                
                # Convert team_id to int for dict lookup
                try:
                    team_id = int(team_id)
                    if name_col not in row or pd.isna(row[name_col]):
                        bracket_df.loc[idx, name_col] = teams_dict.get(team_id, f"Team {team_id}")
                    
                    # Add team abbreviation
                    bracket_df.loc[idx, abbrev_col] = abbrev_dict.get(team_id, f"T{team_id}")
                except:
                    pass
    
    print("\n===== NHL PLAYOFF PREDICTIONS =====\n")
    
    # Group by round
    for round_name in PLAYOFF_ROUNDS:
        round_series = bracket_df[bracket_df['round'] == round_name]
        
        print(f"\n{round_name.upper()}")
        print("-" * 50)
        
        for _, series in round_series.iterrows():
            # Get team identifiers based on preference (abbreviation or full name)
            if pd.isna(series['home_team_id']):
                home_team = "TBD"
            else:
                if use_abbreviations:
                    home_team = series.get('home_team_abbrev')
                    if pd.isna(home_team):
                        team_id = int(series['home_team_id'])
                        home_team = abbrev_dict.get(team_id, f"T{team_id}")
                else:
                    home_team = series.get('home_team_name')
                    if pd.isna(home_team):
                        team_id = int(series['home_team_id'])
                        home_team = teams_dict.get(team_id, f"Team {team_id}")
            
            if pd.isna(series['away_team_id']):
                away_team = "TBD"
            else:
                if use_abbreviations:
                    away_team = series.get('away_team_abbrev')
                    if pd.isna(away_team):
                        team_id = int(series['away_team_id'])
                        away_team = abbrev_dict.get(team_id, f"T{team_id}")
                else:
                    away_team = series.get('away_team_name')
                    if pd.isna(away_team):
                        team_id = int(series['away_team_id'])
                        away_team = teams_dict.get(team_id, f"Team {team_id}")
            
            # Series status and prediction
            if series['completed']:
                if pd.isna(series['winner_id']):
                    print(f"Series {series['series_id']}: {home_team} vs {away_team} - No winner determined")
                    continue
                
                winner_id = int(series['winner_id'])
                if use_abbreviations:
                    winner = series.get('winner_abbrev')
                    if pd.isna(winner):
                        winner = abbrev_dict.get(winner_id, f"T{winner_id}")
                else:
                    winner = series.get('winner_name')
                    if pd.isna(winner):
                        winner = teams_dict.get(winner_id, f"Team {winner_id}")
                
                series_score = f"{int(series['home_wins'])}-{int(series['away_wins'])}"
                if winner_id == (series['home_team_id'] if not pd.isna(series['home_team_id']) else 0):
                    print(f"{home_team} defeats {away_team}, {series_score}")
                else:
                    print(f"{away_team} defeats {home_team}, {series_score}")
            else:
                if pd.isna(series['home_team_id']) or pd.isna(series['away_team_id']):
                    print(f"Series {series['series_id']}: {home_team} vs {away_team} - Waiting for teams")
                else:
                    print(f"Series {series['series_id']}: {home_team} vs {away_team} - Prediction pending")
        
    # Print champion if available
    final_row = bracket_df[bracket_df['round'] == "Stanley Cup Final"]
    if not final_row.empty and final_row.iloc[0]['completed'] and not pd.isna(final_row.iloc[0]['winner_id']):
        winner_id = int(final_row.iloc[0]['winner_id'])
        if use_abbreviations:
            champion = final_row.iloc[0].get('winner_abbrev')
            if pd.isna(champion):
                champion = abbrev_dict.get(winner_id, f"T{winner_id}")
        else:
            champion = final_row.iloc[0].get('winner_name')
            if pd.isna(champion):
                champion = teams_dict.get(winner_id, f"Team {winner_id}")
        
        print("\n" + "=" * 50)
        print(f"PREDICTED STANLEY CUP CHAMPION: {champion}")
        print("=" * 50)
    
    print("\nNote: These predictions are for entertainment purposes only.") 