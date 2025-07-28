# Tehran House Price Prediction

A machine learning project to predict house prices in Tehran using data from Divar.ir listings in 1401 (2022). This project leverages features such as area, number of rooms, parking, storage, elevator, and Address to train predictive models, deployed via an interactive Streamlit application.

**Author**: Mohammadreza Sharifi  
**Student ID**: 403135804  
**GitHub**: [Mo-sharifi](https://github.com/Mo-sharifi)  
**Project Repository**: [Divar_house_price](https://github.com/Mo-sharifi/Divar_house_price)  
**Streamlit app**: [Tehran House Price Predictor](https://divarhousepredictor.streamlit.app/)

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Streamlit Application](#streamlit-application)
- [Prerequisites](#prerequisites)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project, developed as part of an academic endeavor, aims to predict house prices in Tehran using a dataset of listings from Divar.ir in 1401 (2022). By processing features like area, number of rooms, parking, storage, elevator, and Address, multiple machine learning models were trained to provide accurate price predictions. The final model is deployed through a user-friendly Streamlit web application, enabling users to estimate house prices interactively.

## Features
- **Data Collection**: Gathered from Divar.ir house listings in Tehran (1401/2022).
- **Data Preprocessing**: Cleaning, encoding categorical variables (e.g., Address), and normalizing numerical features.
- **Machine Learning Models**: Implementation of Decision Tree, Random Forest, ElasticNet, and XGBoost algorithms.
- **Model Evaluation**: Performance assessed using metrics like RMSE and R².
- **Interactive Interface**: A Streamlit app for seamless house price predictions.


## Dataset
The dataset consists of Tehran house listings from Divar.ir in 1401 (2022) with the following features:
- **Area**: Size of the house in square meters.
- **Number of Rooms**: Number of bedrooms.
- **Parking**: Binary feature (1 for Yes, 0 for No).
- **Storage**: Binary feature (1 for Yes, 0 for No).
- **Elevator**: Binary feature (1 for Yes, 0 for No).
- **Address**: Categorical feature representing the house's location in Tehran.
- **Price**: Target variable (house price in IRR).

Additionally, the dataset included a column for house prices in USD. However, as the exchange rate in 1401 (2022) was approximately 30,000 IRR per USD, this column was excluded from the analysis to maintain consistency with IRR-based pricing.

Preprocessing steps include handling missing values, one-hot encoding for categorical variables (e.g., Address), and normalizing numerical features.

## Models
The project implements and compares the following machine learning models:
- **Decision Tree**: A baseline model for interpretable predictions.
- **Random Forest**: An ensemble model for improved accuracy.
- **ElasticNet**: A linear model with L1 and L2 regularization.
- **XGBoost**: A gradient boosting model for high performance.

## Model Performance

The performance of each model was evaluated using common regression metrics: **R²**, **RMSE**, and **MAE**. The table below summarizes the test set performance for each algorithm:

| Model                | R² Score (Train) | R² Score (Test) | RMSE (IRR)      | MAE (IRR)       |
|----------------------|------------------|------------------|------------------|------------------|
| ElasticNet           | 82.68%           | 80.36%           | 1,122,654,208.31 | 776,652,948.66   |
| Decision Tree        | 83.35%           | 78.56%           | 1,172,796,570.04 | 738,901,760.27   |
| Random Forest        | 88.78%           | 82.48%           | 1,060,303,936.91 | 634,092,430.24   |
| XGBoost              | 88.78%           | 83.43%           | 1,030,961,893.25 | 622,108,130.48   |



>  *XGBoost achieved the highest accuracy on the test set and is therefore used as the final deployed model.*


## Installation
To set up the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Mo-sharifi/Divar_house_price.git
   cd Divar_house_price
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Load Data**:
   Load the dataset using the `load_data.py` script, which imports the house price data from Divar.ir listings (1401/2022) and prepares it for subsequent steps such as preprocessing and model training:
   ```bash
   python src/load_data.py
   ```
2. **Preprocess Data**:
   Process the dataset using the `preprocess_data.py` script, which handles missing values by removing them, converts relevant columns (e.g., transforming the area column into numerical format), removes outliers in the area and price columns using the Interquartile Range (IQR) method, and saves the cleaned dataset to the `/data/preprocess/` directory:
   ```bash
   python src/data_preprocessing.py
   ```
3. **Visualization**:
   Generate five visualizations using the preprocessed dataset with the data_viz.py script to explore patterns and distributions in the data. Run the script to view the plots:
   ```bash
   python src/data_viz.py
   ```
4. **Train Models**:
 Train the machine learning models using the model_trainer.py script. This script employs GridSearchCV to tune hyperparameters for Decision Tree, Random Forest, ElasticNet, and XGBoost algorithms. A pipeline is created that first encodes the Address column using CatBoostEncoder, followed by scaling the data. Each algorithm’s parameters are defined in separate functions, and the script outputs R² scores for both train and test sets, along with MAE and RMSE metrics for the test set. The trained models are saved for later use:
 
 **Why CatBoostEncoder?**  
The Address column contains approximately 180 unique values, making One-Hot Encoding impractical due to the creation of a sparse dataset. Label Encoding was also unsuitable, as it assigns arbitrary numerical values (e.g., 1 for Shahrak-e Gharb, 250 for Molavi), which could mislead the model into assuming an incorrect value hierarchy (e.g., Molavi being "more valuable" than Shahrak-e Gharb). Leave-One-Out Encoding was considered but discarded due to generating NaN values for unseen addresses, which could disrupt the model. CatBoostEncoder was chosen as the optimal solution because it:

-   Provides meaningful encodings for new, unseen data.
-   Prevents data leakage.
-   Performs well on unbalanced datasets or those with rare values, making it ideal for production deployment
```bash
   python src/model_trainer.py
   ```

## Streamlit Application
The project includes an interactive Streamlit web application for predicting house prices. To run the app:
```bash
streamlit run app.py
```

This launches a local web server (typically at `http://localhost:8501`). Users can input house features to receive instant price predictions.

The application is also deployed on Streamlit Cloud, allowing anyone to use the model without any setup:

🔗 [Tehran House Price Predictor](https://divarhousepredictor.streamlit.app/)

## Prerequisites
- Python 3.8 or higher
- Dependencies (listed in `requirements.txt`):
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - xgboost
  - streamlit

## Contributing
Contributions are warmly welcomed! If you have suggestions for improving or optimizing the project, please share them—I’m eager to learn and make this work even better! To contribute::
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Create a Pull Request.

Please ensure your code adheres to the project's style guidelines.

## License
This project is licensed under the [MIT License](LICENSE).

