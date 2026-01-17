# Taxi Fare Prediction

A machine learning project that predicts taxi fare amounts based on pickup/dropoff locations, passenger count, and temporal features using XGBoost Random Forest Regressor.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Het415/Taxi_fare_prediction/blob/main/Taxi_Fair_Prediction.ipynb)

## Overview

This project implements a predictive model for estimating taxi fares using historical trip data. The model leverages geospatial features, temporal patterns, and passenger information to provide accurate fare predictions with an R² score of 0.86.

## Features

- **Geospatial Analysis**: Utilizes pickup and dropoff coordinates to calculate trip distance using the Haversine formula
- **Feature Engineering**: Extracts temporal features including year, month, day, hour, and day of week from timestamp data
- **Advanced ML Model**: Implements XGBoost Random Forest Regressor for robust predictions
- **Comprehensive EDA**: Includes visualizations for fare distribution, distance relationships, and temporal patterns
- **Data Preprocessing**: Handles outliers and ensures data quality through statistical filtering

## Dataset

The dataset contains the following features:
- `amount`: Fare amount (target variable)
- `date_time_of_pickup`: Timestamp of trip start
- `longitude_of_pickup`: Pickup location longitude
- `latitude_of_pickup`: Pickup location latitude
- `longitude_of_dropoff`: Dropoff location longitude
- `latitude_of_dropoff`: Dropoff location latitude
- `no_of_passenger`: Number of passengers

## Tech Stack

**Languages & Libraries:**
- Python 3.x
- pandas - Data manipulation and analysis
- NumPy - Numerical computing
- Matplotlib & Seaborn - Data visualization
- scikit-learn - Machine learning utilities and metrics
- XGBoost - Gradient boosting framework

**Environment:**
- Jupyter Notebook / Google Colab

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Het415/Taxi_fare_prediction.git
cd Taxi_fare_prediction
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

3. Open the notebook:
```bash
jupyter notebook Taxi_Fair_Prediction.ipynb
```

## Project Structure

```
Taxi_fare_prediction/
│
├── Taxi_Fair_Prediction.ipynb    # Main Jupyter notebook with complete analysis
├── README.md                      # Project documentation
└── data/                          # Dataset directory (if applicable)
```

## Methodology

### 1. Data Preprocessing
- Loaded and explored the taxi trip dataset
- Checked for missing values and data types
- Converted timestamp to datetime format
- Identified and filtered outliers based on distance and fare distribution

### 2. Feature Engineering
- **Distance Calculation**: Computed Haversine distance between pickup and dropoff coordinates
- **Temporal Features**: Extracted year, month, day, hour, and day of week from pickup timestamp
- **Feature Selection**: Selected relevant features for model training

### 3. Exploratory Data Analysis
- Analyzed fare amount distribution
- Visualized relationships between distance and fare
- Examined temporal patterns in taxi usage
- Identified correlation between features

### 4. Model Development
- Split data into training and testing sets (80/20)
- Trained XGBoost Random Forest Regressor
- Evaluated model performance using multiple metrics

### 5. Model Evaluation
- R² Score: 0.86 (86% variance explained)
- Mean Absolute Error (MAE): 2.37
- Mean Squared Error (MSE): 24.34
- Root Mean Squared Error (RMSE): 4.93

## Results

The XGBoost Random Forest Regressor achieved strong performance with an R² score of 0.86, indicating the model explains 86% of the variance in taxi fares. The RMSE of 4.93 suggests the model's predictions are typically within ~$5 of the actual fare amount.

**Key Insights:**
- Distance is the strongest predictor of fare amount
- Temporal features provide additional predictive power
- The model effectively handles the non-linear relationship between features and fare

## Usage

To make predictions with the trained model:

```python
# Load the model and prepare your data
import xgboost
import pandas as pd

# Example prediction
sample_data = {
    'year': [2023],
    'month': [6],
    'day': [15],
    'hour': [17],
    'day_of_week': [3],
    'distance': [5.2],
    'no_of_passenger': [1]
}

df = pd.DataFrame(sample_data)
predicted_fare = model.predict(df)
print(f"Predicted Fare: ${predicted_fare[0]:.2f}")
```

## Future Improvements

- Incorporate additional features such as weather conditions, traffic data, and surge pricing
- Experiment with ensemble methods combining multiple models
- Implement hyperparameter tuning for optimal performance
- Add real-time prediction API
- Include pickup zone and dropoff zone categorical features
- Analyze seasonal fare variations in detail

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is open source and available under the [MIT License](LICENSE).

## Contact

Het - [GitHub Profile](https://github.com/Het415)

Project Link: [https://github.com/Het415/Taxi_fare_prediction](https://github.com/Het415/Taxi_fare_prediction)

## Acknowledgments

- Dataset source: NYC Taxi Trip Data
- Inspiration from Kaggle taxi fare prediction competitions
- XGBoost documentation and community
