## Predictive Analytics for Employee Retention Using Ensemble Learning Models

This project predicts employee attrition using an end-to-end, production-style machine learning pipeline built with **Python**, **Scikit-learn**, **XGBoost**, and **SQL-friendly data ingestion**.  
The goal is to provide HR teams with actionable risk scores and interpretable drivers of attrition for strategic workforce planning.

### Key capabilities

- **XGBoost classification with automated hyperparameter tuning**
  - Uses an `XGBClassifier` wrapped in a Scikit-learn `Pipeline`.
  - Hyperparameters (depth, learning rate, estimators, subsample, etc.) are tuned via `RandomizedSearchCV` on a stratified train split.
  - On the included IBM HR attrition dataset, the tuned model achieves **~0.85 test accuracy** and **~0.83 F1-score**, consistent with the results reported in the notebook.
  - The pipeline automatically extracts and ranks the **top 15+ attrition risk factors** using model feature importances.

- **Automated feature engineering and data quality validation**
  - `attrition_pipeline.py` adds engineered features such as:
    - prior experience (`years_before_company`)
    - compensation normalized by tenure (`income_per_year_at_company`)
    - early-career flags, etc.
  - Uses vectorized Pandas operations and Scikit-learn transformers (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`) for consistent preprocessing.
  - A simple but explicit validation step computes a **data quality score** (fraction of rows passing range and null checks), which is typically **≥ 0.99** on the provided dataset.

- **Efficient, scalable ETL for 100K+ employee records**
  - ETL and preprocessing are implemented in a fully vectorized way (no Python row loops).
  - `benchmark_etl` in `attrition_pipeline.py` programmatically scales the dataset to **~100K rows** and times end-to-end preprocessing.
  - This design mirrors the workflow used to obtain **multi‑x speedups** over a naive, non-vectorized baseline by leveraging parallel, optimized numerical kernels.

### Files

- `HR_Employee_Attrition.csv` – IBM HR attrition dataset used for training and evaluation.
- `EAPM.ipynb` – original exploratory notebook with EDA, baselines, and model comparisons (Logistic Regression, Random Forest, SVM, XGBoost, etc.).
- `attrition_pipeline.py` – production-style pipeline with:
  - feature engineering and validation
  - XGBoost + hyperparameter tuning
  - feature-importance based risk factor ranking
  - ETL benchmarking for large (100K+) synthetic employee datasets.
- `requirements.txt` – Python dependencies.

### How to run

```bash
pip install -r requirements.txt
python attrition_pipeline.py
```

This will:

- load and validate the HR attrition dataset  
- engineer additional features  
- run XGBoost with automated hyperparameter tuning  
- report accuracy, F1-score, confusion matrix, and classification report  
- print the top 15+ most important attrition drivers  
- benchmark vectorized ETL on a ~100K-row synthetic dataset.
