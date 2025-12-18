import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def twelve_month_rf(df):

    TARGET = "event_status"

    # drop IDs, leakage, and all 6m features
    DROP_COLS = [
        "person_id",
        "death_date",
        "last_activity_date",
        "last_of_death_or_visit",
        "first_bone_event_date",
        "T_ref",
    ] + [c for c in df.columns if "_6m" in c]

    # gender → binary
    df = df.copy()
    df["gender_binary"] = (
        df["gender_concept_name"]
        .astype(str)
        .str.lower()
        .map({"male": 1, "female": 0})
    )

    DROP_COLS.append("gender_concept_name")

    FEATURE_COLS = [
        c for c in df.columns
        if c not in DROP_COLS + [TARGET]
    ]

    X = df[FEATURE_COLS]
    y = df[TARGET].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), FEATURE_COLS)
    ])

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_auc = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc"
    )

    print("CV AUC per fold:", cv_auc)
    print("Mean CV AUC:", cv_auc.mean())

    pipeline.fit(X_train, y_train)

    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_proba)

    print("Hold-out test AUC:", test_auc)
