
# Hyperparameter Optimization of Machine Learning Models using Reinforcement Learning  
**Case Study: Tehran Stock Exchange**

## Overview
This project explores an **automated framework for hyperparameter optimization** of machine learning models using **Reinforcement Learning (RL)**.  
The proposed method leverages **Proximal Policy Optimization (PPO)** to tune hyperparameters dynamically and intelligently, aiming to enhance predictive performance in **stock price forecasting** within the Tehran Stock Exchange.

---

## Key Objectives
- Develop a reinforcement learning framework for automated hyperparameter tuning.  
- Apply the framework to several ML models:  
  - Linear Regression  
  - Random Forest  
  - Gradient Boosting  
  - XGBoost  
  - LightGBM  

---

## Methodology
1. **Data Collection & Preprocessing**  
   - Stock price data from Tehran Stock Exchange.  
   - Cleaning, normalization, and feature engineering (returns, volatility, Sharpe ratio, drawdowns, etc.).  
   - Stationarity tests (ADF, KPSS).  

2. **Modeling**  
   - Implementation of ML models for price prediction.  

3. **Reinforcement Learning Approach**  
   - Agent: PPO-based optimizer.  
   - Environment: ML models and evaluation metrics.  
   - Reward Function: Based on predictive performance (MSE).  

---

## Results
- RL significantly improved prediction accuracy compared to default and traditional tuning methods.  
- Demonstrated robustness of RL in handling noisy and volatile financial data.  

---

## Project Structure
```
project-root/
│── src/                            # Source code (data_fetching, data preprocessing, modeling, RL agent)
│   ├── data_fetching.py           
│   ├── preprocessing.py
│   ├── models.py
│   ├── rl_agent.py
│   └── training_pipeline.py
│── notebooks/                      # Jupyter notebooks for experiments & analysis
│   ├── Main_pipeline.ipynb+
│── environment.yml 
│── LICENSE
│── README.md

---

## Installation

To run this project, it is recommended to use **Conda** to recreate the exact environment used in development. Follow the steps below:

1. **Clone the repository:**
```bash
git clone https://github.com/MohammadGholami-MG/RL-HyperOpt-TSE.git
cd RL-HyperOpt-TSE
2. **Create the Conda environment:**
conda env create -f environment.yml
3. **Activate the environment:**
conda activate rl-hyperopt-tse_env

---

## Usage
Run preprocessing and training:
```bash
python src/training_pipeline.py

```

Or open the Jupyter notebooks:
```bash
jupyter notebook notebooks/Main_pipeline.ipynb

```

## Technologies Used
- **Python 3.10+**
- **Scikit-learn**
- **XGBoost / LightGBM**
- **Stable-Baselines3 (PPO)**
- **Pandas, NumPy, Matplotlib, Seaborn**

---

## Future Work
- Extend RL-based optimization to **deep learning models (LSTM, Transformers)**.    
- Investigate **multi-agent RL** for collaborative optimization of multiple models.  

---

## Author
**Mohammad Gholami**  
LinkedIn: https://www.linkedin.com/in/mohammad-gholami-mgh22

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

