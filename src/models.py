import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error



# Train the model and evaluate performance on test data
def train_and_evaluate(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)               # Train model
    y_pred = model.predict(x_test)            # Predict on test data
    r2 = r2_score(y_test, y_pred)             # Calculate R-squared (accuracy)
    mse = mean_squared_error(y_test, y_pred)  # Calculate Mean Squared Error (error)
    return {"model": model, "r2": r2, "mse": mse, "y_pred": y_pred}

# Run multiple models on all processed datasets
def run_models_on_processed_data(processed_data):
    all_results = {}  # Store results for each (symbol, feature)

    # Define models to test
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "LightGBM": lgb.LGBMRegressor(n_estimators=100, random_state=42)
    }

    # Loop through each symbol and feature dataset
    for (symbol, feature), data_dict in processed_data.items():
        x_train, y_train = data_dict["x_train"], data_dict["y_train"]
        x_test, y_test = data_dict["x_test"], data_dict["y_test"]
        results = {}
        # Train and evaluate each model
        for name, model in models.items():
            results[name] = train_and_evaluate(model, x_train, y_train, x_test, y_test)

        all_results[(symbol, feature)] = results

    return all_results 



def results_to_dataframe(all_results):
    rows = []
    for (symbol, feature), model_results in all_results.items():
        for model_name, metrics in model_results.items():
            rows.append({
                "Symbol": symbol,
                "Feature": feature,
                "Model": model_name,
                "R2_Score": metrics["r2"],
                "MSE": metrics["mse"]
            })
    df_results = pd.DataFrame(rows)
    return df_results



def get_min_error_per_feature(all_results, metric='mse'):        # metric= 'mse' 
    records = []

    for (symbol, feature), results in all_results.items():
        best_model = None
        best_metric_value = None

        for model_name, result in results.items():
            value = result[metric]
            if best_metric_value is None or value < best_metric_value:
                best_metric_value = value
                best_model = model_name

        records.append({
            "Symbol": symbol,
            "Feature": feature,
            "Best_Model": best_model,
            f"Min_{metric}": best_metric_value
        })

    df = pd.DataFrame(records)
    return df



def plot_predictions(y_test, y_pred, model_name, symbol, feature):
    plt.figure(figsize=(20, 5))
    plt.plot(y_test, label='Actual', color='blue')
    plt.plot(y_pred, label='Predicted', color='red', alpha=0.7)
    plt.title(f'Prediction vs Actual for {model_name} - {symbol} - {feature}')
    plt.xlabel('Samples')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    