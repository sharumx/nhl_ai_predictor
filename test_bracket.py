import logging
import pandas as pd
from src.utils import load_dataframe
from src.playoff_prediction import print_playoff_bracket, setup_playoff_bracket

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_bracket')

def main():
    try:
        # Try to load the existing bracket
        try:
            bracket_df = load_dataframe('playoff_bracket_20242025.csv', processed=True)
            logger.info("Loaded existing bracket")
        except:
            # Create a new bracket if needed
            logger.info("Creating new bracket")
            bracket_df = setup_playoff_bracket('20242025')
        
        # Simulate a few series to have some completed ones
        # For simplicity, we'll just mark some series as completed with random winners
        import random
        for i in range(4):  # Complete 4 series in the first round
            idx = i  # First 4 series
            if not pd.isna(bracket_df.loc[idx, 'home_team_id']) and not pd.isna(bracket_df.loc[idx, 'away_team_id']):
                winner = random.choice([bracket_df.loc[idx, 'home_team_id'], bracket_df.loc[idx, 'away_team_id']])
                bracket_df.loc[idx, 'winner_id'] = winner
                bracket_df.loc[idx, 'completed'] = True
                if winner == bracket_df.loc[idx, 'home_team_id']:
                    bracket_df.loc[idx, 'home_wins'] = 4
                    bracket_df.loc[idx, 'away_wins'] = random.randint(0, 3)
                else:
                    bracket_df.loc[idx, 'away_wins'] = 4
                    bracket_df.loc[idx, 'home_wins'] = random.randint(0, 3)
        
        # Print the bracket with abbreviations
        print_playoff_bracket(bracket_df)
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    main() 