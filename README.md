# Workflow-CI

Repository CI/CD untuk re-training model Heart Disease menggunakan MLflow Project + GitHub Actions.

## Struktur Folder

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions workflow
└── MLProject/
    ├── MLProject                     # MLflow Project definition
    ├── conda.yaml                    # Environment dependencies
    ├── modelling.py                  # Training script
    ├── DockerHub.txt                 # Tautan Docker Hub image
    └── heart_preprocessing/          # Dataset hasil preprocessing
        ├── X_train.csv
        ├── X_test.csv
        ├── y_train.csv
        └── y_test.csv
```

## GitHub Secrets yang Diperlukan

Tambahkan secrets berikut di **Settings → Secrets and variables → Actions**:

| Secret | Keterangan |
|---|---|
| `MLFLOW_TRACKING_USERNAME` | Username DagsHub |
| `MLFLOW_TRACKING_PASSWORD` | Token DagsHub |
| `DOCKER_USERNAME` | Username Docker Hub |
| `DOCKER_PASSWORD` | Password/Token Docker Hub |

## Cara Trigger

Workflow berjalan otomatis saat push ke branch `main` yang menyentuh folder `MLProject/`, atau bisa di-trigger manual via tab **Actions → Run workflow**.

## Docker Hub

Image tersedia di: https://hub.docker.com/r/bimzt/heart-disease-mlflow