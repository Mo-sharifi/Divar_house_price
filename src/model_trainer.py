# === Standard Library ===
import os 
import time  

# === Third-Party Libraries ===
import pandas as pd  
import numpy as np  
import joblib  # For saving/loading models

# === Scikit-learn ===
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    KFold,
) # Model selection utilities
from sklearn.preprocessing import StandardScaler  # Feature scaling
from sklearn.linear_model import ElasticNet  # ElasticNet regression
from sklearn.ensemble import RandomForestRegressor  # Random Forest model
from sklearn.tree import DecisionTreeRegressor  # Decision Tree model
from sklearn.pipeline import Pipeline  # Creating ML pipelines
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)  # Evaluation metrics

# === Other Libraries ===
import category_encoders as ce  # Encoding categorical variables
from xgboost import XGBRegressor  # XGBoost model

# === Custom Modules ===
from data_loader import data_load  # Function to load data
from data_preprocessing import preprocess_data  # Function to clean and preprocess data

# Set global float display format
pd.options.display.float_format = "{:,.0f}".format


def create_model(df, model, parameters):
    """
    Trains a regression model using a pipeline with LeaveOneOutEncoder and StandardScaler.
    Uses GridSearchCV for hyperparameter tuning, evaluates the model, and saves the best model.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with features and target column 'Price'.
    - model (estimator): A scikit-learn compatible regression model.
    - parameters (dict): Hyperparameter grid for GridSearchCV.
    """
    # Split the dataset into training and test sets
    X = df.drop(columns=["Price"])
    y = df["Price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    start = time.time()  # Start timer for training duration

    # Build pipeline: encoding categorical features, scaling, and applying the model
    pipeline = Pipeline(
        [
            ("encoder", ce.LeaveOneOutEncoder(cols=["Address"])),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )

    # Perform hyperparameter tuning with cross-validation
    grid = GridSearchCV(
        pipeline,
        parameters,
        cv=KFold(n_splits=5, shuffle=True, random_state=123),
        scoring="r2",
        n_jobs=-1,
    )

    # Train the model
    grid_fit = grid.fit(X_train, y_train)

    # Make predictions
    y_train_pred = grid_fit.predict(X_train)
    y_pred = grid_fit.predict(X_test)

    # Evaluate performance
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    MAE = mean_absolute_error(y_test, y_pred)

    model_name = str(model).split("(")[0]  # Get model name
    end = time.time()  # End timer

    # Print evaluation results
    print(f"The best parameters for {model_name} model is: {grid_fit.best_params_}")
    print("--" * 10)
    print(f"(R2 score) in the training set is {r2_train:0.2%} for {model_name} model.")
    print(f"(R2 score) in the testing set is {r2_test:0.2%} for {model_name} model.")
    print(f"RMSE is {RMSE:,} for {model_name} model.")
    print(f"MAE is {int(MAE):,} for {model_name} model.")
    print("--" * 20)
    print(f"Runtime of the program is: {end - start:0.3f} seconds")

    # Save the trained model
    print("\nSaving model! ")
    time.sleep(3)
    save_model(grid_fit.best_estimator_, model_name)
    print(f"Model {model_name} was saved !!!!")


def save_model(model, model_name, directory="Divar_house_prediction/models"):
    """
    Saves the trained model to disk in the specified directory.

    Parameters:
    - model: Trained model object to be saved.
    - model_name (str): Name of the model (used for filename).
    - directory (str): Target directory to save the model file.
    """
    os.makedirs(directory, exist_ok=True)
    filename = f"{model_name}.sav"
    path = os.path.join(directory, filename)
    joblib.dump(model, path)
    print(f"Model saved at: {path}")



def random_forest_regressor():
    """
    Returns a RandomForestRegressor model and corresponding hyperparameter grid.
    """
    rfr = RandomForestRegressor(random_state=123)
    param = {
        "model__n_estimators": [50, 100, 150],
        "model__max_depth": [4, 6, 8],
        "model__min_samples_split": [3, 5, 7],
        "model__min_samples_leaf": [2, 3],
    }
    return rfr, param

def main() -> None:
    """
    Main function to run the model training pipeline for multiple regressors.
    """
    df = data_load()  # Load raw data
    clean_data = preprocess_data(df)  # Clean and preprocess data

    print("Random Forest\n")
    create_model(
        clean_data,
        model=random_forest_regressor()[0],
        parameters=random_forest_regressor()[1],
    )



if __name__ == "__main__":
    main()
