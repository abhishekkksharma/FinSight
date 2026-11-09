# FinSight — Complete Project Overview & Guide

Welcome to **FinSight**! This guide is written for anyone completely new to the project, AI, or software automation. It explains what FinSight is, how all the pieces work together, what every single file in the project does, and how the code operates under the hood.

---

## 1. The Big Picture: What is FinSight?

### The Problem It Solves
Imagine you are planning to buy a car or take out a monthly loan (called an **EMI** — Equated Monthly Installment). Before taking on a new monthly expense of \$15,000, you need to answer a critical financial question:  
> *"Can I actually afford this every month without running out of money or draining my savings?"*

To figure this out, you need to know how much money you spend each month on life essentials (rent, groceries, utilities, transit, insurance). However, many people don't keep track of every penny, or they want a realistic estimate based on their profile.

### The Solution
**FinSight** is an intelligent financial assistant that:
1. **Predicts Living Expenses**: It takes simple personal details — like your Income, Age, Number of Dependents, Occupation, and City Tier — and uses **Artificial Intelligence / Machine Learning (ML)** to estimate your total monthly living expenses.
2. **Evaluates Loan Affordability**: It subtracts your predicted living expenses and your proposed loan payment (EMI) from your monthly income.
3. **Gives a Safety Verdict**: It tells you whether the loan is **Affordable**, **Tight**, or **Not Affordable**, ensuring you still keep a safety net (by default, saving at least 20% of your income).

---

## 2. How FinSight Works: The 3-Layer Architecture

Think of FinSight like a modern restaurant:

```
[ User on Web Browser ]  <--- Internet --->  [ Backend API (Server) ]  <--->  [ AI Brain (ML Model) ]
  (Frontend Dashboard)                        (FastAPI Web Server)             (Trained Python Pipeline)
```

1. **The Brain (Machine Learning Engine)**: This part reads historical data from thousands of financial profiles, discovers spending patterns, and creates a mathematical model that can guess anyone's monthly expenses.
2. **The Waiter / Bridge (Backend API)**: Powered by Python and FastAPI, this server takes input from the user, asks the AI Brain for an expense estimate, calculates whether the user can afford their loan, and sends back the final results.
3. **The Storefront (Frontend Dashboard)**: An interactive web dashboard built with React and TailwindCSS where users type in their income and loan details, click a button, and immediately see clean visual cards showing their financial outcome.

---

## 3. Project File Map: What Every Major File Does

Here is a simple breakdown of every folder and file in the project workspace:

```
FinSight/
├── data/
│   └── dataset.csv                 <-- The historical database of financial profiles
├── models/
│   └── best_model.pkl              <-- The saved, trained AI "Brain"
├── outputs/                        <-- Charts, accuracy statistics, and tuning rules
│   ├── metrics.json
│   ├── model_comparison.csv
│   ├── expense_calibration.json
│   └── *.png (charts & graphs)
├── notebooks/
│   └── EDA.ipynb                   <-- Data exploration playground for analysis
├── src/                            <-- Python Backend & Machine Learning source code
│   ├── utils.py                    <-- Helper functions & file path manager
│   ├── preprocess.py               <-- Data cleaner & transformer
│   ├── evaluate.py                 <-- Quality inspector & chart generator
│   ├── train.py                    <-- AI model trainer & selector
│   ├── predict.py                  <-- Expense estimator & affordability calculator
│   └── api.py                      <-- Web server endpoints (FastAPI)
├── frontend/finsight/              <-- React Web Application
│   ├── src/App.tsx                 <-- The visual user dashboard screen
│   ├── src/main.tsx                <-- Web app entry launcher
│   └── package.json / configs      <-- Frontend dependencies and setup
├── methodology.md                  <-- Project methodology documentation
├── README.md                       <-- Quick-start setup guide
└── requirements.txt                <-- Python package list
```

---

### Detailed File Guide

#### A. Data & Models Storage
* **[dataset.csv](file:///c:/Users/Angadveer%20Singh/Downloads/FinSight/data/dataset.csv)**: Contains rows of historical financial data (Income, Age, Dependents, Occupation, City Tier) along with individual expenses (Rent, Groceries, Transport, Insurance, Loan Repayment).
* **`models/best_model.pkl`**: The saved binary file containing the trained machine learning pipeline. Think of it as frozen intelligence ready to make predictions instantly without retraining.

#### B. Machine Learning & Backend Source Code (`src/`)
* **[utils.py](file:///c:/Users/Angadveer%20Singh/Downloads/FinSight/src/utils.py)** (*The Helper Toolkit*):
  * Sets up standard file paths across the project.
  * Manages logging (printing helpful status messages to the console).
  * Saves JSON files safely.
* **[preprocess.py](file:///c:/Users/Angadveer%20Singh/Downloads/FinSight/src/preprocess.py)** (*The Data Cleaner & Transformer*):
  * **Cleans data**: Fixes spelling mistakes (e.g., converting "Engineer" to "Professional", "tier 1" to "Tier_1"), removes duplicate rows, and filters out invalid numbers (like negative age or negative income).
  * **Avoids Target Leakage**: Calculates `Total_Expense` by summing up Rent + Groceries + Transport + Insurance + Loan Repayment, but removes those individual breakdown columns from input features so the model doesn't cheat!
  * **Builds Preprocessing Pipeline**: Converts text inputs (like Occupation) into 0s and 1s (One-Hot Encoding) and scales numbers (Income, Age) so algorithms can understand them.
  * **Behavioral Calibration**: Calculates adjustments for subtle factors like age brackets and number of family dependents.
* **[evaluate.py](file:///c:/Users/Angadveer%20Singh/Downloads/FinSight/src/evaluate.py)** (*The Quality Inspector & Chart Creator*):
  * Computes mathematical accuracy scores: **RMSE** (Root Mean Squared Error), **MAE** (Mean Absolute Error), and **R²** (R-squared score showing accuracy).
  * Generates visual diagnostic graphs: prediction accuracy scatter plots, error residual graphs, feature importance charts, and learning curves.
* **[train.py](file:///c:/Users/Angadveer%20Singh/Downloads/FinSight/src/train.py)** (*The AI Model Trainer*):
  * Trains multiple regression models (Linear Regression, Ridge, ElasticNet, Decision Trees, Random Forests, Gradient Boosting, XGBoost, LightGBM, CatBoost).
  * Compares all models on a test set of data.
  * Smartly selects a model that provides **smooth, realistic predictions** (preferring regularized models like Ridge Regression so small input changes don't cause wild price jumps).
  * Fine-tunes the winning model's settings using 5-fold cross-validation (`GridSearchCV`) and saves the fitted pipeline to `models/best_model.pkl`.
* **[predict.py](file:///c:/Users/Angadveer%20Singh/Downloads/FinSight/src/predict.py)** (*The Expense Predictor & Financial Advisor*):
  * Loads the saved model from disk.
  * Accepts user inputs (Income, Age, Dependents, Occupation, City Tier) and outputs predicted monthly expenses.
  * Contains `assess_affordability()`: Calculates remaining monthly surplus after expenses and loan EMI, comparing it against the required monthly savings buffer (default 20%).
  * Can be run directly from the command line interface (CLI).
* **[api.py](file:///c:/Users/Angadveer%20Singh/Downloads/FinSight/src/api.py)** (*The Server Communication Hub*):
  * Built using **FastAPI**.
  * Defines two endpoints:
    * `/health`: Checks if the web server is online.
    * `/predict`: Receives JSON data from the web dashboard, validates input formats (using Pydantic), calculates expense & affordability, and sends JSON responses back to the dashboard.

#### C. User Interface / Frontend (`frontend/finsight/`)
* **[App.tsx](file:///c:/Users/Angadveer%20Singh/Downloads/FinSight/frontend/finsight/src/App.tsx)** (*The Dashboard Screen*):
  * A modern web interface built in React and styled with TailwindCSS.
  * Displays two main input cards: **Financial Profile** (Income, Age, Dependents, Occupation, City Tier) and **EMI Planning** (Monthly EMI, Loan Tenure, Minimum Savings Buffer %).
  * Sends an HTTP POST request to `http://127.0.0.1:8000/predict` when the user clicks "Check affordability".
  * Renders interactive response cards: **Predicted Expense**, **Affordability Verdict** (Affordable / Tight / Not Affordable), **Surplus After EMI**, and **Total EMI Commitment**.

---

## 4. How the Code Works & Executes Step-by-Step

### Phase 1: Machine Learning Model Training (Offline Pipeline)
```
Raw Data (dataset.csv) ➔ Clean & Validate ➔ Feature Transformer ➔ Model Competition ➔ Tuning & Calibration ➔ Save best_model.pkl
```
1. `train.py` reads `dataset.csv` via `preprocess.py`.
2. Missing values are filled, text categories normalized, and `Total_Expense` target created.
3. The dataset is split 80% for training and 20% for testing.
4. Multiple ML algorithms compete. Ridge Regression is selected for smooth financial behavior.
5. Hyperparameters are tuned to maximize prediction accuracy.
6. Diagnostic plots and accuracy metrics are saved to `outputs/`.
7. The complete preprocessing + model pipeline is serialized to `models/best_model.pkl`.

### Phase 2: Live Prediction & User Interaction (Online Flow)
```
User Enters Input on Web Dashboard (App.tsx)
              │
              ▼ (HTTP POST /predict)
   FastAPI Web Server (api.py)
              │
              ├──> 1. Normalize Categories & Run ML Brain (predict.py) ──> Predicted Living Expense
              │
              └──> 2. Perform Financial Affordability Logic ───────────────> Verdict & Surplus Math
              │
              ▼ (JSON Response)
User Sees Instant Visual Results on Dashboard!
```

#### The Affordability Math Formula
When a user inputs an income of \$75,000 and a proposed EMI of \$15,000:
$$\text{Current Surplus} = \text{Income} - \text{Predicted Expense}$$
$$\text{Surplus After EMI} = \text{Current Surplus} - \text{Monthly EMI}$$
$$\text{Required Buffer} = \text{Income} \times \text{Minimum Savings Rate (e.g., 20\%)}$$

* **Affordable**: If $\text{Surplus After EMI} \ge \text{Required Buffer}$
* **Tight**: If $0 \le \text{Surplus After EMI} < \text{Required Buffer}$ (Payment can be made, but savings buffer is sacrificed)
* **Not Affordable**: If $\text{Surplus After EMI} < 0$ (Income is less than expenses plus EMI)

---

## 5. Summary & Next Steps

FinSight seamlessly bridges data science, backend API engineering, and frontend web development into a unified product.

Whenever you want to break down a specific file's code line-by-line, we can explore that file in depth!
