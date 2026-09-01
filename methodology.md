# Project Methodology: FinSight - Expense Prediction & Affordability Assessment

This document outlines the step-by-step methodology adopted for developing the **FinSight** project, which includes a machine learning model for predicting personal expenses and an affordability assessment tool, exposed via a backend API and an interactive frontend interface.

## 1. Objective and Problem Statement
The primary goal of this project is to build an end-to-end intelligent system that accurately predicts an individual's monthly expenses based on demographic and financial factors. By doing so, it helps assess whether a user can comfortably afford an additional financial burden, such as a new loan EMI, without compromising their minimum required savings rate.

## 2. Step-by-Step Methodology

### Phase 1: Data Collection and Preprocessing
1. **Data Ingestion**: Raw data containing financial and demographic details (income, age, dependents, occupation, city tier, etc.) is loaded.
2. **Data Cleaning**: Handling missing values, standardizing string representations, and dropping duplicate or irrelevant records.
3. **Feature Engineering & Target Creation**: Creating the target variable `Expense` if not directly available, and establishing baseline financial rules (e.g., extracting or estimating historical expenses).
4. **Behavioral Calibration**: Capturing relationships between different demographics (e.g., city tiers, occupations) and their impact on spending habits to adjust baseline expectations.
5. **Outlier Detection**: Using the Interquartile Range (IQR) method to identify and handle extreme values in numerical features (like Income and Expense).
6. **Data Transformation (Pipeline)**: Building a robust Scikit-Learn preprocessing pipeline to scale numerical features and encode categorical variables for model compatibility.

### Phase 2: Model Training and Evaluation
1. **Train-Test Split**: The dataset is split into training (80%) and testing (20%) sets to ensure unbiased evaluation of the model's generalization capability.
2. **Model Selection**: Several regression algorithms are trained and compared (see section 3).
3. **Evaluation Metrics**: Models are evaluated using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²).
4. **Model Selection Strategy**: The best model is chosen not only based on raw RMSE but also by prioritizing model families that provide "smooth" and stable predictions (e.g., Ridge, Linear Regression, or ensemble trees) when user inputs slightly change.
5. **Hyperparameter Tuning**: The selected model family is tuned using 5-fold Cross-Validation (`GridSearchCV`) to find the optimal hyperparameters.
6. **Model Persisting**: The final, best-performing tuned model pipeline (which includes the preprocessor) is serialized and saved using `joblib` for future inference.

### Phase 3: Analysis and Visualization
1. **Exploratory Data Analysis (EDA)**: Generating correlation heatmaps, distribution plots, and boxplots to understand feature distributions.
2. **Model Diagnostics**: Plotting actual vs. predicted values, residual plots, and learning curves to diagnose bias/variance.
3. **Feature Importance**: Extracting and saving the relative importance of different features (like Income, Age, City Tier) in predicting expenses.

### Phase 4: API Development (Backend)
1. **Framework Selection**: Using **FastAPI** to create a robust and high-performance REST API.
2. **Endpoint Creation**: 
    - `/health`: For basic health checks.
    - `/predict`: Accepts user data (Income, Age, Dependents, EMI details, etc.), runs it through the pre-loaded ML pipeline to predict expenses, and calculates affordability (e.g., checking if `Income - Predicted Expense - EMI >= Minimum Savings`).
3. **Data Validation**: Enforcing input data validation and constraints (e.g., positive income, realistic age ranges) using **Pydantic** models.

### Phase 5: Frontend Development (User Interface)
1. **Framework**: A modern single-page application built with **React** (via **Vite**) and **TypeScript**.
2. **Styling**: Using **TailwindCSS** for rapid, responsive, and aesthetic UI design.
3. **Integration**: Creating forms to capture user inputs, making HTTP POST requests to the FastAPI backend, and dynamically rendering the predicted expense and affordability status on the dashboard.

---

## 3. Machine Learning Models Evaluated
During the training phase, multiple regression algorithms were built, evaluated, and compared:
- **Linear Regression**: A simple, interpretable baseline.
- **Ridge & ElasticNet Regression**: Linear models with L2/L1 regularization to prevent overfitting and handle multicollinearity.
- **Decision Tree Regressor**: A non-linear model capturing complex feature interactions.
- **Random Forest Regressor**: An ensemble of decision trees that reduces variance and improves robustness.
- **Gradient Boosting Regressor**: A sequential ensemble method that iteratively corrects errors of previous models.
- **Optional Advanced Models**: Support for XGBoost, LightGBM, and CatBoost (if installed in the environment).

---

## 4. Technology Stack & Libraries Used

### Machine Learning & Data Science (Python)
- **`pandas` & `numpy`**: For data manipulation, numerical operations, and tabular data structures.
- **`scikit-learn`**: For preprocessing pipelines, model training, cross-validation, and evaluation metrics.
- **`matplotlib` & `seaborn`**: For generating comprehensive EDA and model evaluation visualizations.
- **`joblib`**: For saving and loading the trained Scikit-Learn pipelines.

### Backend API (Python)
- **`FastAPI`**: A modern, fast (high-performance) web framework for building APIs.
- **`uvicorn`**: An ASGI web server implementation for Python used to serve the FastAPI application.
- **`pydantic`**: For strict data validation and type hinting in API requests and responses.

### Frontend Application (JavaScript/TypeScript)
- **`React (v19)`**: A JavaScript library for building component-based user interfaces.
- **`Vite`**: A fast, modern build tool and development server.
- **`TypeScript`**: For adding static typing to JavaScript, improving code quality and maintainability.
- **`TailwindCSS`**: A utility-first CSS framework for styling the web application effortlessly.

---

## 5. Conclusion
This systematic approach ensures that the FinSight project is built on clean data, utilizes robust machine learning practices, and is delivered through a modern, scalable, and user-friendly web architecture. The separation of concerns (ML pipeline -> Backend API -> Frontend UI) makes the project highly maintainable and extensible for future enhancements.
