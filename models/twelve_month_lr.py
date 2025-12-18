import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def twelve_month_lr(df):

    TARGET = "event_status"

    # dynamically drop all 6m features
    DROP_COLS = [
        "person_id",
        "death_date",
        "last_activity_date",
        "last_of_death_or_visit",
        "first_bone_event_date",
        "T_ref",
    ] + [c for c in df.columns if "_6m" in c]

    # encode gender as binary
    df = df.copy()
    df["gender_binary"] = (
        df["gender_concept_name"]
        .astype(str)
        .str.lower()
        .map({"male": 1, "female": 0})
    )

    DROP_COLS.append("gender_concept_name")

    # everything else numeric
    NUMERIC_COLS = [
        c for c in df.columns
        if c not in DROP_COLS + [TARGET]
    ]

    X = df.drop(columns=DROP_COLS + [TARGET])
    y = df[TARGET].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_COLS)
    ])

    model = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        max_iter=1000
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
