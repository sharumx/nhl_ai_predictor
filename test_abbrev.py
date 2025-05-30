import pandas as pd

# Load teams data
teams_df = pd.read_csv('data/raw/teams.csv')

# Print team data with abbreviations
print("Team ID\tName\tAbbreviation")
print("-" * 40)
for _, team in teams_df.iterrows():
    print(f"{team['team_id']}\t{team['team_name']}\t{team['team_abbrev']}")

# Create a simulated playoff series and print it with abbreviations
print("\nSimulated Playoff Series:")
print("-" * 40)

# Sample series
series = [
    {"home": 1, "away": 8, "score": "4-2"}, 
    {"home": 2, "away": 7, "score": "4-3"},
    {"home": 3, "away": 6, "score": "4-1"},
    {"home": 4, "away": 5, "score": "3-4"}
]

# Create lookup dictionaries
name_dict = dict(zip(teams_df['team_id'], teams_df['team_name']))
abbrev_dict = dict(zip(teams_df['team_id'], teams_df['team_abbrev']))

# Print series with team names and abbreviations
print("With Full Names:")
for match in series:
    home_name = name_dict.get(match['home'], f"Team {match['home']}")
    away_name = name_dict.get(match['away'], f"Team {match['away']}")
    if match['score'].startswith('4'):
        print(f"{home_name} defeats {away_name}, {match['score']}")
    else:
        print(f"{away_name} defeats {home_name}, {match['score'].split('-')[1]}-{match['score'].split('-')[0]}")

print("\nWith Abbreviations:")
for match in series:
    home_abbrev = abbrev_dict.get(match['home'], f"T{match['home']}")
    away_abbrev = abbrev_dict.get(match['away'], f"T{match['away']}")
    if match['score'].startswith('4'):
        print(f"{home_abbrev} defeats {away_abbrev}, {match['score']}")
    else:
        print(f"{away_abbrev} defeats {home_abbrev}, {match['score'].split('-')[1]}-{match['score'].split('-')[0]}") 