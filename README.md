# Expense Prediction Model

This project trains a regression model to estimate a user's expected monthly expenses from demographic and financial profile features. The saved model is intended for later integration into a financial affordability application.

The model predicts `Total_Expense`, created as:

```text
Rent + Loan_Repayment + Insurance + Groceries + Transport
```

Those five expense columns are intentionally excluded from the model inputs to avoid target leakage.

## Dataset

The training data is stored at `data/dataset.csv`.

Input features:

- `Income`
- `Age`
- `Dependents`
- `Occupation`
- `City_Tier`

Target:

- `Total_Expense`

## Folder Structure

```text
expense_prediction/
├── data/
│   └── dataset.csv
├── models/
│   └── best_model.pkl
├── notebooks/
│   └── EDA.ipynb
├── outputs/
│   ├── metrics.json
│   ├── model_comparison.csv
│   ├── feature_importance.png
│   ├── residual_plot.png
│   ├── prediction_vs_actual.png
│   └── learning_curve.png
├── src/
│   ├── evaluate.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── train.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

Optional model libraries are detected automatically if installed:

- XGBoost
- LightGBM
- CatBoost

## Train the Model

From the `expense_prediction` directory:

```bash
python src/train.py
```

Training performs data validation, cleaning, EDA plot generation, model comparison, hyperparameter tuning, final evaluation, and model saving.

## Prediction Example

```bash
python src/predict.py --income 50000 --age 30 --dependents 2 --occupation Professional --city-tier Tier_2
```

The prediction script accepts common aliases such as `Engineer`, which is mapped to `Professional`, and city tier spellings such as `Tier-1`, `Tier_1`, or `1`.

Supported training categories:

- Occupation: `Professional`, `Retired`, `Self_Employed`, `Student`
- City tier: `Tier_1`, `Tier_2`, `Tier_3`

## Affordability Example

To check whether a user can afford a `15000` monthly car EMI for 5 years:

```bash
python src/predict.py --income 75000 --age 25 --dependents 2 --occupation Engineer --city-tier Tier-1 --emi 15000 --years 5
```

The script predicts monthly expenses, subtracts the proposed EMI, and checks whether the remaining surplus still meets the required monthly buffer. The default buffer is `20%` of income and can be changed:

```bash
python src/predict.py --income 75000 --age 25 --dependents 2 --occupation Engineer --city-tier Tier-1 --emi 15000 --years 5 --min-savings-rate 0.15
```

You can also import the function:

```python
from src.predict import assess_affordability, predict_expense

expense = predict_expense(
    income=50000,
    age=30,
    dependents=2,
    occupation="Professional",
    city_tier="Tier_2",
)

affordability = assess_affordability(
    income=50000,
    predicted_expense=expense,
    monthly_emi=15000,
    tenure_years=5,
)
```

## Evaluation Outputs

The training script writes:

- `outputs/metrics.json`
- `outputs/model_comparison.csv`
- `outputs/prediction_vs_actual.png`
- `outputs/residual_plot.png`
- `outputs/feature_importance.png`
- `outputs/learning_curve.png`
- EDA plots for income, age, occupation, city tier, dependents, target distribution, boxplots, and correlations

## Future Improvements

- Add richer financial behavior features after validating they are available at prediction time.
- Track experiments with MLflow or a similar registry.
- Add model drift checks after production integration.
- Wrap `predict_expense` in a FastAPI service for the affordability application.
