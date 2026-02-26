import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


TARGET_COL = "Attrition"


@dataclass
class PipelineResults:
    accuracy: float
    f1: float
    confusion: np.ndarray
    report: str
    top_features: List[Tuple[str, float]]
    data_quality: float


def load_data(csv_path: str = "HR_Employee_Attrition.csv") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find dataset at {csv_path}")
    return pd.read_csv(csv_path)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "TotalWorkingYears" in df and "YearsAtCompany" in df:
        df["years_before_company"] = (
            df["TotalWorkingYears"] - df["YearsAtCompany"]
        ).clip(lower=0)
    if "MonthlyIncome" in df and "YearsAtCompany" in df:
        denom = df["YearsAtCompany"].replace(0, np.nan)
        df["income_per_year_at_company"] = df["MonthlyIncome"] / denom
        df["income_per_year_at_company"] = df[
            "income_per_year_at_company"
        ].fillna(df["MonthlyIncome"])
    if "Age" in df and "TotalWorkingYears" in df:
        df["early_career_flag"] = (df["TotalWorkingYears"] <= 5).astype(int)
    return df


def compute_data_quality(df: pd.DataFrame) -> float:
    checks = []
    checks.append(df.notna().all(axis=1))
    if "Age" in df:
        checks.append((df["Age"] >= 18) & (df["Age"] <= 65))
    if "MonthlyIncome" in df:
        checks.append(df["MonthlyIncome"] > 0)
    mask = np.logical_and.reduce(checks)
    return mask.mean()


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str]]:
    drop_cols = [
        c
        for c in [
            TARGET_COL,
            "EmployeeNumber",
            "EmployeeCount",
            "Over18",
            "StandardHours",
        ]
        if c in X.columns
    ]
    X = X.drop(columns=drop_cols, errors="ignore")
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    return preprocessor, X.columns.tolist()


def build_xgb_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_estimators=300,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=-1,
    )
    return Pipeline(steps=[("pre", preprocessor), ("model", xgb)])


def tune_xgb_hyperparams(
    pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series
) -> Pipeline:
    param_dist = {
        "model__n_estimators": [200, 300, 400, 500],
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
        "model__subsample": [0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.8, 0.9, 1.0],
        "model__min_child_weight": [1, 3, 5],
        "model__gamma": [0.0, 0.1, 0.3],
    }
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=20,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def extract_feature_importance(
    pipeline: Pipeline, original_features: List[str], top_n: int = 15
) -> List[Tuple[str, float]]:
    model = pipeline.named_steps["model"]
    pre = pipeline.named_steps["pre"]
    if not hasattr(model, "feature_importances_"):
        return []
    importances = model.feature_importances_
    feature_names: List[str] = []
    for name, trans, cols in pre.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "get_feature_names_out"):
            names = trans.get_feature_names_out(cols)
        else:
            names = cols
        feature_names.extend(names.tolist())
    pairs = list(zip(feature_names, importances))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_n]


def run_pipeline() -> PipelineResults:
    df = load_data()
    df = add_engineered_features(df)
    data_quality = compute_data_quality(df)

    y = (df[TARGET_COL] == "Yes").astype(int)
    X = df.drop(columns=[TARGET_COL])

    preprocessor, feature_cols = build_preprocessor(X)
    base_pipe = build_xgb_pipeline(preprocessor)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    tuned_pipe = tune_xgb_hyperparams(base_pipe, X_train, y_train)
    y_pred = tuned_pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    top_features = extract_feature_importance(tuned_pipe, feature_cols, top_n=20)

    return PipelineResults(
        accuracy=acc,
        f1=f1,
        confusion=cm,
        report=report,
        top_features=top_features,
        data_quality=data_quality,
    )


def benchmark_etl(n_target_rows: int = 100_000) -> float:
    df = load_data()
    reps = max(1, int(np.ceil(n_target_rows / len(df))))
    big_df = pd.concat([df] * reps, ignore_index=True).head(n_target_rows)
    big_df = add_engineered_features(big_df)

    y = (big_df[TARGET_COL] == "Yes").astype(int)
    X = big_df.drop(columns=[TARGET_COL])
    preprocessor, _ = build_preprocessor(X)

    start = time.perf_counter()
    _ = preprocessor.fit_transform(X)
    elapsed = time.perf_counter() - start
    return elapsed


if __name__ == "__main__":
    results = run_pipeline()
    print("================ XGBoost with Hyperparameter Tuning ================")
    print(f"Test Accuracy: {results.accuracy:.4f}")
    print(f"Test F1-score: {results.f1:.4f}")
    print("\nConfusion matrix:")
    print(results.confusion)
    print("\nClassification report:")
    print(results.report)
    print(f"\nData quality score: {results.data_quality:.3f}")
    print("\nTop 15+ attrition risk factors (feature importance):")
    for name, score in results.top_features[:15]:
        print(f"{name:40s} {score:.4f}")

    etl_time = benchmark_etl()
    print(
        f"\nVectorized ETL preprocessing time for ~100K rows: "
        f"{etl_time:.2f} seconds"
    )
