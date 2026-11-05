"""Четыре этапа Olist: выгрузка, проверка, подготовка и обучение."""

import csv
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = ["price", "freight", "weight", "n_items", "installments", "est_days"]
CATEGORICAL_COLUMNS = ["customer_state", "seller_state", "category", "payment_type"]
REQUIRED_COLUMNS = [
    "order_id", "order_purchase_timestamp",
    *NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS, "late",
]


def _read_csv(path):
    # Проверяем исходный заголовок: pandas переименовывает повторяющиеся столбцы.
    with Path(path).open(encoding="utf-8", newline="") as file:
        reader = csv.reader(file, strict=True)
        header = next(reader, [])
        if len(header) != len(set(header)):
            raise ValueError("В CSV повторяются названия столбцов")
        missing = sorted(set(REQUIRED_COLUMNS) - set(header))
        if missing:
            raise ValueError(f"В CSV нет обязательных столбцов: {', '.join(missing)}")
        # Иначе pandas может молча принять лишнее поле за индекс строки.
        for row in reader:
            if row and len(row) != len(header):
                raise ValueError(f"Число полей CSV не совпадает с заголовком: строка {reader.line_num}")
        file.seek(0)
        return pd.read_csv(
            file,
            dtype={
                column: str
                for column in ["order_id", "order_purchase_timestamp", *CATEGORICAL_COLUMNS]
            },
            float_precision="round_trip",
        )


def _sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def extract(run_key):
    """Сохранить public.orders в отдельный CSV для запуска."""
    import psycopg

    if not isinstance(run_key, str) or not run_key.strip():
        raise ValueError("run_key должен быть непустой строкой")

    # Хеш даёт безопасное имя каталога для любого Airflow run_id.
    run_directory = hashlib.sha256(run_key.encode("utf-8")).hexdigest()
    directory = Path(os.environ["OLIST_WORKDIR"]).resolve() / "runs" / run_directory
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = directory / "snapshot.csv"
    temporary = directory / "snapshot.tmp"
    database_url = os.environ["DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://", 1)
    query = """
        COPY (
            SELECT order_id, order_purchase_timestamp, price, freight, weight,
                   n_items, installments, est_days, customer_state, seller_state,
                   category, payment_type, late
            FROM public.orders
            ORDER BY order_purchase_timestamp, order_id
        ) TO STDOUT WITH (FORMAT CSV, HEADER TRUE, ENCODING 'UTF8')
    """
    try:
        with psycopg.connect(database_url, connect_timeout=10) as connection:
            with connection.cursor() as cursor:
                with cursor.copy(query) as rows, temporary.open("wb") as file:
                    for block in rows:
                        file.write(block)
        temporary.replace(snapshot)
    finally:
        temporary.unlink(missing_ok=True)
    return str(snapshot)


def validate(snapshot):
    """Проверить локальный CSV без записей, сети и других побочных действий."""
    frame = _read_csv(snapshot)
    if frame.empty:
        raise ValueError("CSV не содержит заказов")

    ids = frame["order_id"]
    if ids.isna().any() or ids.str.strip().eq("").any():
        raise ValueError("order_id не должен быть пустым")
    if ids.duplicated().any():
        raise ValueError("В CSV повторяются order_id")

    for column in [*NUMERIC_COLUMNS, "late"]:
        if pd.api.types.is_bool_dtype(frame[column]):
            raise ValueError(f"{column} должен содержать числа, а не True/False")
        try:
            values = frame[column].to_numpy(dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{column} должен содержать числа") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"В {column} есть пропуски, NaN или Inf")
        if column == "late" and not np.isin(values, [0, 1]).all():
            raise ValueError("late должен содержать только 0 или 1")

    dates = pd.to_datetime(
        frame["order_purchase_timestamp"], format="ISO8601", errors="raise", utc=True,
    )
    if dates.isna().any():
        raise ValueError("order_purchase_timestamp не должен содержать пропуски")
    return str(Path(snapshot).resolve())


def prepare(snapshot):
    """Сохранить временное разбиение и вернуть пути с метаданными снимка."""
    from olist_model import split_by_time

    snapshot = Path(validate(snapshot))
    snapshot_sha256 = _sha256(snapshot)
    cutoff = os.environ.get("OLIST_CUTOFF", "2017-07-31")
    train_frame, validation_frame = split_by_time(_read_csv(snapshot), cutoff=cutoff)
    train_path = snapshot.parent / "train.csv"
    validation_path = snapshot.parent / "validation.csv"
    train_frame.to_csv(train_path, index=False, encoding="utf-8")
    validation_frame.to_csv(validation_path, index=False, encoding="utf-8")
    return {
        "snapshot": str(snapshot),
        "snapshot_sha256": snapshot_sha256,
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "cutoff": cutoff,
        "train_rows": len(train_frame),
        "validation_rows": len(validation_frame),
    }


def train(prepared, *, dag_run_id, epochs=None, seed=None):
    """Обучить готовую модель, записать эксперимент и вернуть MLflow run_id."""
    import mlflow
    from olist_model import fit_model, save_predictor

    if not isinstance(dag_run_id, str) or not dag_run_id.strip():
        raise ValueError("dag_run_id должен быть непустой строкой")
    snapshot = Path(prepared["snapshot"])
    if _sha256(snapshot) != prepared["snapshot_sha256"]:
        raise ValueError("Исходный снимок изменился после prepare")

    train_frame = _read_csv(prepared["train_path"])
    validation_frame = _read_csv(prepared["validation_path"])
    if len(train_frame) != prepared["train_rows"] or len(validation_frame) != prepared["validation_rows"]:
        raise ValueError("Размеры выборок не совпадают с результатом prepare")

    epochs = int(os.environ.get("OLIST_EPOCHS", "5")) if epochs is None else epochs
    seed = int(os.environ.get("OLIST_SEED", "42")) if seed is None else seed
    lr = float(os.environ.get("OLIST_LR", "0.003"))
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "olist-delivery"))
    with mlflow.start_run(tags={"dag_run_id": dag_run_id, "snapshot_path": str(snapshot)}) as run:
        mlflow.log_params({
            "epochs": epochs, "seed": seed, "lr": lr,
            "cutoff": prepared["cutoff"],
            "train_rows": len(train_frame),
            "validation_rows": len(validation_frame),
            "snapshot_sha256": prepared["snapshot_sha256"],
        })
        mlflow.log_artifact(str(snapshot), artifact_path="data")
        predictor, history = fit_model(train_frame, validation_frame, epochs=epochs, seed=seed, lr=lr)
        for row in history:
            mlflow.log_metrics({
                "train_loss": row["train_loss"],
                "validation_pr_auc": row["validation_pr_auc"],
            }, step=row["epoch"])
        model_directory = snapshot.parent / "models" / run.info.run_id
        save_predictor(predictor, model_directory)
        mlflow.log_artifacts(str(model_directory), artifact_path="model")
    return run.info.run_id
