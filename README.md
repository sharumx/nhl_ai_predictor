# NHL AI Predictor

A machine learning system that predicts NHL game outcomes, including win probabilities, score predictions, and playoff bracket simulations.

## Overview

NHL AI Predictor uses historical NHL game data to train machine learning models that can predict:

- The winner of upcoming NHL games
- The expected score for each team
- The total score (over/under)
- Full playoff bracket simulations, including round-by-round advancement and Stanley Cup champion

The system uses a combination of team statistics, recent performance trends, and head-to-head matchups to make its predictions.

## Features

- Data collection from NHL API
- Automatic data preprocessing and feature engineering
- Machine learning models (Random Forest) for win and score prediction
- Playoff bracket simulation with series predictions
- Command-line interface for predictions
- RESTful API for accessing predictions programmatically
- Support for offline operation with sample data

## Directory Structure

```
nhl_ai_predictor/
│
├── data/                    # Data storage
│   ├── raw/                 # Raw data from API
│   └── processed/           # Processed data for modeling
│
├── models/                  # Trained model files
│
├── src/                     # Source code
│   ├── data_collection.py   # Data collection from NHL API
│   ├── preprocessing.py     # Data preprocessing and feature engineering
│   ├── model.py             # Model training and prediction
│   ├── playoff_prediction.py # Playoff bracket simulations
│   ├── api.py               # Flask API for predictions
│   └── utils.py             # Utility functions
│
├── notebooks/               # Jupyter notebooks for analysis
│
├── main.py                  # Main application entry point
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Installation

1. Clone the repository:

```
git clone https://github.com/yourusername/nhl_ai_predictor.git
cd nhl_ai_predictor
```

2. Create a virtual environment (optional but recommended):

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```
pip install -r requirements.txt
```

## Usage

### Command Line Interface

The system provides a command-line interface for generating predictions:

#### Game Predictions

```
# Run the full pipeline (collect data, preprocess, train models, predict)
python main.py

# Specify NHL seasons to analyze
python main.py --seasons 20222023 20212022

# Predict games for the next N days
python main.py --days 14

# Skip model retraining
python main.py --no-retrain

# Only collect data (no preprocessing or prediction)
python main.py --collect-only

# Only preprocess data (no prediction)
python main.py --preprocess-only

# Only train models (no prediction)
python main.py --train-only
```

#### Playoff Predictions

```
# Simulate the entire playoff bracket and predict the Stanley Cup champion
python main.py --playoff

# Specify a season for playoff predictions
python main.py --playoff --playoff-season 20232024

# Simulate only a specific playoff round
python main.py --playoff --playoff-round "First Round"
python main.py --playoff --playoff-round "Second Round"
python main.py --playoff --playoff-round "Conference Finals"
python main.py --playoff --playoff-round "Stanley Cup Final"

# Adjust the number of simulations per series for more accurate results
python main.py --playoff --simulations 5000
```

### API Server

The system also provides a RESTful API for accessing predictions:

```
# Start the API server
python -m src.api
```

API Endpoints:

- `GET /api/health` - Health check
- `GET /api/predictions?days=7` - Get predictions for upcoming games
- `GET /api/teams` - Get NHL teams information
- `GET /api/predictions/latest` - Get the most recent predictions

### Offline Operation

If you encounter issues with the NHL API, the system can operate with sample data:

1. The system will automatically generate sample data if it cannot connect to the NHL API
2. You can modify the sample data in `src/utils.py` to create custom test scenarios

## Model Performance

The system uses Random Forest models for both win and score predictions:

- Win prediction accuracy: ~60-65% (varies by season)
- Score prediction RMSE: ~1.0-1.5 goals (varies by season)
- Playoff series prediction: Each playoff series is simulated multiple times (default: 1000) to account for randomness and provide the most likely outcome

Performance metrics are logged during model training and can be viewed in the console output.

## Playoff Prediction Methodology

The playoff prediction system works as follows:

1. Creates a playoff bracket based on team standings (points)
2. For each playoff matchup, simulates a best-of-7 series multiple times
3. Accounts for home-ice advantage using the 2-2-1-1-1 NHL playoff format
4. Advances winners to the next round until a Stanley Cup champion is determined
5. Displays results including series outcomes (e.g., "Team A defeats Team B, 4-2")

The system can simulate:

- First Round (8 series)
- Second Round (4 series)
- Conference Finals (2 series)
- Stanley Cup Final (1 series)

## Contributing

Contributions are welcome! Here are some ways you can contribute:

- Add new features (e.g., player-based predictions, betting odds integration)
- Improve model accuracy
- Enhance the API functionality
- Fix bugs and improve error handling
- Improve documentation

## Troubleshooting

If you encounter issues with the NHL API:

1. Check your internet connection
2. Verify that the NHL API endpoints are correct (they may change over time)
3. Use the sample data functionality to test the system without API access

If models aren't performing well:

1. Check the processed data quality with `python main.py --preprocess-only`
2. Try collecting more historical data with `python main.py --collect-only --seasons 20222023 20212022 20202021`
3. Review the feature engineering in `preprocessing.py`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NHL API for providing access to hockey data
- Scikit-learn for machine learning tools
- Flask for the API framework

## Disclaimer

This tool is for entertainment purposes only. Predictions should not be used for betting or financial decisions.
