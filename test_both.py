import os
import logging
import pandas as pd
import random
from src.utils import load_dataframe, save_dataframe
from src.playoff_prediction import print_playoff_bracket, setup_playoff_bracket

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_bracket')

def create_test_bracket():
    """Create a simple test bracket with some completed series"""
    # Load teams data
    teams_df = load_dataframe('teams.csv', processed=False)
    
    # Create a simple bracket structure
    rounds = ["First Round", "Second Round", "Conference Finals", "Stanley Cup Final"]
    bracket = []
    
    # First round: 8 series
    # Eastern Conference
    bracket.append({
        'round': 'First Round', 'series_id': 'E1', 
        'home_team_id': 1, 'away_team_id': 8, 
        'home_wins': 4, 'away_wins': 2, 
        'winner_id': 1, 'completed': True,
        'season': '20242025'
    })
    bracket.append({
        'round': 'First Round', 'series_id': 'E2', 
        'home_team_id': 5, 'away_team_id': 7, 
        'home_wins': 4, 'away_wins': 3, 
        'winner_id': 5, 'completed': True,
        'season': '20242025'
    })
    bracket.append({
        'round': 'First Round', 'series_id': 'E3', 
        'home_team_id': 2, 'away_team_id': 4, 
        'home_wins': 1, 'away_wins': 4, 
        'winner_id': 4, 'completed': True,
        'season': '20242025'
    })
    bracket.append({
        'round': 'First Round', 'series_id': 'E4', 
        'home_team_id': 6, 'away_team_id': 3, 
        'home_wins': 3, 'away_wins': 4, 
        'winner_id': 3, 'completed': True,
        'season': '20242025'
    })
    
    # Western Conference
    bracket.append({
        'round': 'First Round', 'series_id': 'W1', 
        'home_team_id': 9, 'away_team_id': 12, 
        'home_wins': 4, 'away_wins': 1, 
        'winner_id': 9, 'completed': True,
        'season': '20242025'
    })
    bracket.append({
        'round': 'First Round', 'series_id': 'W2', 
        'home_team_id': 13, 'away_team_id': 16, 
        'home_wins': 2, 'away_wins': 4, 
        'winner_id': 16, 'completed': True,
        'season': '20242025'
    })
    bracket.append({
        'round': 'First Round', 'series_id': 'W3', 
        'home_team_id': 11, 'away_team_id': 14, 
        'home_wins': 4, 'away_wins': 3, 
        'winner_id': 11, 'completed': True,
        'season': '20242025'
    })
    bracket.append({
        'round': 'First Round', 'series_id': 'W4', 
        'home_team_id': 10, 'away_team_id': 15, 
        'home_wins': 2, 'away_wins': 4, 
        'winner_id': 15, 'completed': True,
        'season': '20242025'
    })
    
    # Second round: 4 series
    bracket.append({
        'round': 'Second Round', 'series_id': 'E1', 
        'home_team_id': 1, 'away_team_id': 5, 
        'home_wins': 4, 'away_wins': 1, 
        'winner_id': 1, 'completed': True,
        'season': '20242025'
    })
    bracket.append({
        'round': 'Second Round', 'series_id': 'E2', 
        'home_team_id': 4, 'away_team_id': 3, 
        'home_wins': 3, 'away_wins': 4, 
        'winner_id': 3, 'completed': True,
        'season': '20242025'
    })
    bracket.append({
        'round': 'Second Round', 'series_id': 'W1', 
        'home_team_id': 9, 'away_team_id': 16, 
        'home_wins': 4, 'away_wins': 3, 
        'winner_id': 9, 'completed': True,
        'season': '20242025'
    })
    bracket.append({
        'round': 'Second Round', 'series_id': 'W2', 
        'home_team_id': 11, 'away_team_id': 15, 
        'home_wins': 2, 'away_wins': 4, 
        'winner_id': 15, 'completed': True,
        'season': '20242025'
    })
    
    # Conference Finals: 2 series
    bracket.append({
        'round': 'Conference Finals', 'series_id': 'E', 
        'home_team_id': 1, 'away_team_id': 3, 
        'home_wins': 4, 'away_wins': 2, 
        'winner_id': 1, 'completed': True,
        'season': '20242025'
    })
    bracket.append({
        'round': 'Conference Finals', 'series_id': 'W', 
        'home_team_id': 9, 'away_team_id': 15, 
        'home_wins': 1, 'away_wins': 4, 
        'winner_id': 15, 'completed': True,
        'season': '20242025'
    })
    
    # Stanley Cup Final: 1 series
    bracket.append({
        'round': 'Stanley Cup Final', 'series_id': 'F', 
        'home_team_id': 1, 'away_team_id': 15, 
        'home_wins': 4, 'away_wins': 3, 
        'winner_id': 1, 'completed': True,
        'season': '20242025'
    })
    
    # Create DataFrame
    bracket_df = pd.DataFrame(bracket)
    
    # Add team names and abbreviations
    teams_dict = dict(zip(teams_df['team_id'], teams_df['team_name']))
    abbrev_dict = dict(zip(teams_df['team_id'], teams_df['team_abbrev']))
    
    bracket_df['home_team_name'] = bracket_df['home_team_id'].map(teams_dict)
    bracket_df['away_team_name'] = bracket_df['away_team_id'].map(teams_dict)
    bracket_df['home_team_abbrev'] = bracket_df['home_team_id'].map(abbrev_dict)
    bracket_df['away_team_abbrev'] = bracket_df['away_team_id'].map(abbrev_dict)
    bracket_df['winner_name'] = bracket_df['winner_id'].map(teams_dict)
    bracket_df['winner_abbrev'] = bracket_df['winner_id'].map(abbrev_dict)
    
    return bracket_df

def main():
    try:
        # Create a test bracket
        bracket_df = create_test_bracket()
        
        # Print with full names
        print("\n\n==== PLAYOFF BRACKET WITH FULL TEAM NAMES ====")
        print_playoff_bracket(bracket_df, use_abbreviations=False)
        
        # Print with abbreviations
        print("\n\n==== PLAYOFF BRACKET WITH TEAM ABBREVIATIONS ====")
        print_playoff_bracket(bracket_df, use_abbreviations=True)
        
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main() 