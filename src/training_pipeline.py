import pandas as pd

from data_fetching import import_data
from preprocessing import calculate_data_index, plot_data_index, adf_test, kpss_test, data_preprocessing
from models import run_models_on_processed_data, results_to_dataframe, get_min_error_per_feature, plot_predictions
from rl_agent import optimize_rf_with_rl, optimize_xgb_with_rl, optimize_gbt_with_rl, optimize_lgb_with_rl 


data = import_data(  symbols_input = ['شپنا','شاخص کل','خودرو']    # Enter symbols
                   , symbols_start_date = '1403-01-01'              # Enter start date
                   , symbols_end_date = '1403-12-30')               # Enter end date

with pd.ExcelWriter("Initial_Data.xlsx", engine="xlsxwriter") as writer:
    for sheet_name, df in data.items():
        df.to_excel(writer, sheet_name=sheet_name)

data_index = calculate_data_index(data,
                              window=20,                  # window is depends on your goal 
                              data_price_type='Close',    # (short-term trading,standard,long-term investing)(5,20,100)
                              risk_free_rate=0.0)         # data_price_type : 'Open'   'Close'  'High'   'Low'
                                                          # for daily risk-free rate,The risk-free rate is, 
                                                          # the "baseline return" you could get with no risk,
                                                          # It’s used in Sharpe Ratio to compare your return ,
                                                          # above that safe baseline.

print(data_index)

plot_data_index(data_index, symbol='شپنا'
                          , column='Close'
                          , color='black' )


# Store processed data for ML
processed_data = {}

# Store which features passed stationarity per symbol
stationary_features_by_symbol = {}

# List of features (indexes) to check
columns_to_test = ['simple_return', 'log_return', 'cumulative_return',
                   'rolling_avg_return', 'rolling_volatility', 'rolling_sharpe',
                   'rolling_max', 'max_drawdown', 'drawdown']

# Main loop
for symbol, df in data_index.items():
    stationary_features = []

    for column in columns_to_test:
        if column in df.columns:
            series = df[column].dropna()
            is_adf_stationary = adf_test(series)
            is_kpss_stationary = kpss_test(series)

            if is_adf_stationary or is_kpss_stationary:
                stationary_features.append(column)

                # Preprocess for ML and store
                x_train, y_train, x_test, y_test = data_preprocessing(series.values,
                                                                     num_lags=10,
                                                                     train_test_split=0.8)

                processed_data[(symbol, column)] = {
                    "x_train": x_train,
                    "y_train": y_train,
                    "x_test": x_test,
                    "y_test": y_test
                }

    # Save which features are stationary for this symbol
    stationary_features_by_symbol[symbol] = stationary_features

print(stationary_features_by_symbol)

print(processed_data)


all_results = run_models_on_processed_data(processed_data)
df_results = results_to_dataframe(all_results)
df_results.to_excel("model_results.xlsx", index=False, sheet_name="Model_Results")
print(df_results)

df_min_errors = get_min_error_per_feature(all_results, metric='mse')    # metric= 'mse' 
print(df_min_errors)


print(stationary_features_by_symbol)     


symbol = 'شپنا'     
feature = 'simple_return'

for model_name, results in all_results[(symbol, feature)].items():
    y_test = processed_data[(symbol, feature)]["y_test"]
    y_pred = results["y_pred"]
    plot_predictions(y_test, y_pred, model_name, symbol, feature)



# RandomForest 
# Run the optimization and store results
best_rf_results = optimize_rf_with_rl(processed_data, timesteps=20000)

# Convert to DataFrame
df_rf = pd.DataFrame([
    {
        "Symbol": s,
        "Feature": f,
        "Best_n_estimators": d["best_n_estimators"],
        "Best_max_depth": d["best_max_depth"],
        "Best_MSE": d["best_mse"]
    }
    for (s, f), d in best_rf_results.items()
])

# Save to Excel
df_rf.to_excel("rf_results.xlsx", index=False, sheet_name="RF_Results")

# Sort results by best MSE
print(df_rf.sort_values(by="Best_MSE"))



# XGBoost 
# Run the Optimization and Store Results in DataFrame 
best_xgb_results = optimize_xgb_with_rl(processed_data, timesteps=20000)

# Convert to DataFrame
df_xgb = pd.DataFrame([
    {
        "Symbol": s,
        "Feature": f,
        "Best_n_estimators": d["best_n_estimators"],
        "Best_max_depth": d["best_max_depth"],
        "Best_learning_rate": d["best_learning_rate"],
        "Best_MSE": d["best_mse"]
    }
    for (s, f), d in best_xgb_results.items()
])
# Save to Excel
df_xgb.to_excel("xgb_results.xlsx", index=False, sheet_name="XGB_Results")

# Sort results by best MSE (lower is better)
print(df_xgb.sort_values(by="Best_MSE"))



# GradientBoosting
# Execute Optimization and Aggregate Results 
best_gbt_results = optimize_gbt_with_rl(processed_data, timesteps=20000)

# Convert to DataFrame
df_gbt = pd.DataFrame([
    {
        "Symbol": s,
        "Feature": f,
        "Best_n_estimators": d["best_n_estimators"],
        "Best_max_depth": d["best_max_depth"],
        "Best_learning_rate": d["best_learning_rate"],
        "Best_MSE": d["best_mse"]
    }
    for (s, f), d in best_gbt_results.items()
])

# Save to Excel
df_gbt.to_excel("gbt_results.xlsx", index=False, sheet_name="GBT_Results")

# Display sorted results
print(df_gbt.sort_values(by="Best_MSE"))



# LightGBM
# Run Optimization and Format Results 
best_lgb_results = optimize_lgb_with_rl(processed_data, timesteps=20000)

# Convert results to a pandas DataFrame
df_lgb = pd.DataFrame([
    {
        "Symbol": s,
        "Feature": f,
        "Best_num_leaves": d["best_num_leaves"],
        "Best_max_depth": d["best_max_depth"],
        "Best_learning_rate": d["best_learning_rate"],
        "Best_MSE": d["best_mse"]
    }
    for (s, f), d in best_lgb_results.items()
])

# Display sorted results
print(df_lgb.sort_values(by="Best_MSE"))

# Save to Excel
df_lgb.to_excel("lgb_results.xlsx", index=False, sheet_name="LightGBM_Results")