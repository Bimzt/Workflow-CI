import os
import json
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators',      type=int,   default=100)
parser.add_argument('--max_depth',         type=int,   default=10)
parser.add_argument('--min_samples_split', type=int,   default=2)
parser.add_argument('--max_features',      type=str,   default='sqrt')
args = parser.parse_args()

# Credentials dari environment
MLFLOW_USERNAME = os.environ.get('MLFLOW_TRACKING_USERNAME')
MLFLOW_PASSWORD = os.environ.get('MLFLOW_TRACKING_PASSWORD')

if not MLFLOW_USERNAME or not MLFLOW_PASSWORD:
    raise ValueError("MLFLOW credentials tidak ditemukan di environment variables")

os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_PASSWORD

mlflow.set_tracking_uri("https://dagshub.com/Bimzt/Eksperimen_SML_Bima_Setia.mlflow")

# Load data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'heart_preprocessing')

X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
X_test  = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv')).squeeze()
y_test  = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv')).squeeze()

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# Helper: artefak
def save_confusion_matrix(y_true, y_pred, path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Penyakit Jantung'],
                yticklabels=['Normal', 'Penyakit Jantung'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix - Random Forest')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path

def save_feature_importance(model, feature_names, path='feature_importance.png'):
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices][::-1],
            color='#2196F3', edgecolor='black', alpha=0.8)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
    ax.set_xlabel('Importance Score')
    ax.set_title('Top 15 Feature Importance - Random Forest')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path

def save_classification_report(y_true, y_pred, path='classification_report.json'):
    report = classification_report(y_true, y_pred, output_dict=True)
    with open(path, 'w') as f:
        json.dump(report, f, indent=4)
    return path

# Training
env_run_id = os.environ.get('MLFLOW_RUN_ID')

if env_run_id:
    print(f"Menggunakan run dari CLI: {env_run_id}")
    with mlflow.start_run(run_id=env_run_id):
        params = {
            'n_estimators'     : args.n_estimators,
            'max_depth'        : args.max_depth,
            'min_samples_split': args.min_samples_split,
            'max_features'     : args.max_features,
            'random_state'     : 42
        }
        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred      = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        cv_scores   = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

        acc     = accuracy_score(y_test, y_pred)
        f1      = f1_score(y_test, y_pred)
        prec    = precision_score(y_test, y_pred)
        rec     = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_prob)

        mlflow.log_metric("accuracy",         acc)
        mlflow.log_metric("f1_score",         f1)
        mlflow.log_metric("precision",        prec)
        mlflow.log_metric("recall",           rec)
        mlflow.log_metric("roc_auc",          roc_auc)
        mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
        mlflow.log_metric("cv_accuracy_std",  cv_scores.std())

        mlflow.sklearn.log_model(model, artifact_path="model")

        cm_path  = save_confusion_matrix(y_test, y_pred)
        fi_path  = save_feature_importance(model, X_train.columns.tolist())
        rep_path = save_classification_report(y_test, y_pred)

        mlflow.log_artifact(cm_path,  artifact_path="plots")
        mlflow.log_artifact(fi_path,  artifact_path="plots")
        mlflow.log_artifact(rep_path, artifact_path="reports")

        run_id = mlflow.active_run().info.run_id
else:
    mlflow.set_experiment("heart-disease-ci")
    with mlflow.start_run(run_name="RandomForest_CI"):
        params = {
            'n_estimators'     : args.n_estimators,
            'max_depth'        : args.max_depth,
            'min_samples_split': args.min_samples_split,
            'max_features'     : args.max_features,
            'random_state'     : 42
        }
        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred      = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        cv_scores   = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

        acc     = accuracy_score(y_test, y_pred)
        f1      = f1_score(y_test, y_pred)
        prec    = precision_score(y_test, y_pred)
        rec     = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_prob)

        mlflow.log_metric("accuracy",         acc)
        mlflow.log_metric("f1_score",         f1)
        mlflow.log_metric("precision",        prec)
        mlflow.log_metric("recall",           rec)
        mlflow.log_metric("roc_auc",          roc_auc)
        mlflow.log_metric("cv_accuracy_mean", cv_scores.mean())
        mlflow.log_metric("cv_accuracy_std",  cv_scores.std())

        mlflow.sklearn.log_model(model, artifact_path="model")

        cm_path  = save_confusion_matrix(y_test, y_pred)
        fi_path  = save_feature_importance(model, X_train.columns.tolist())
        rep_path = save_classification_report(y_test, y_pred)

        mlflow.log_artifact(cm_path,  artifact_path="plots")
        mlflow.log_artifact(fi_path,  artifact_path="plots")
        mlflow.log_artifact(rep_path, artifact_path="reports")

        run_id = mlflow.active_run().info.run_id

print(f"\nRun ID: {run_id}")
print(f"   Accuracy : {acc:.4f}")
print(f"   F1 Score : {f1:.4f}")
print(f"   ROC-AUC  : {roc_auc:.4f}")

# Simpan run_id
with open(os.path.join(BASE_DIR, '..', 'run_id.txt'), 'w') as f:
    f.write(run_id)

for fp in [cm_path, fi_path, rep_path]:
    if os.path.exists(fp):
        os.remove(fp)

print("\nTraining selesai.")