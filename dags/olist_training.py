"""Еженедельное обучение модели Olist."""

from datetime import datetime, timedelta, timezone

from airflow.sdk import dag, get_current_context, task


@dag(
    schedule="@weekly",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    default_args={"execution_timeout": timedelta(minutes=5)},
    tags=["olist"],
)
def olist_training():
    @task(retries=2, retry_delay=timedelta(seconds=30))
    def extract():
        from olist_pipeline import extract as extract_orders

        return extract_orders(get_current_context()["run_id"])

    @task(retries=0, trigger_rule="all_success")
    def validate(snapshot):
        from olist_pipeline import validate as validate_orders

        return validate_orders(snapshot)

    @task(retries=0, trigger_rule="all_success")
    def prepare(snapshot):
        from olist_pipeline import prepare as prepare_orders

        return prepare_orders(snapshot)

    @task(retries=0, trigger_rule="all_success", execution_timeout=timedelta(minutes=10))
    def train(prepared):
        from olist_pipeline import train as train_model

        return train_model(prepared, dag_run_id=get_current_context()["run_id"])

    snapshot = extract()
    checked_snapshot = validate(snapshot)
    prepared = prepare(checked_snapshot)
    train(prepared)


olist_training()
