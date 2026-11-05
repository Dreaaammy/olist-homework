"""Проверки данных без PostgreSQL и сервера MLflow."""

import hashlib
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from olist_pipeline import NUMERIC_COLUMNS, REQUIRED_COLUMNS, prepare, train, validate


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.snapshot = Path(self.directory.name) / "snapshot.csv"
        self.frame = pd.DataFrame({
            "order_id": ["001", "1", "003", "004"],
            "order_purchase_timestamp": [
                "2017-07-30T12:00:00", "2017-07-30T13:00:00",
                "2017-07-31T00:00:00", "2017-08-01T12:00:00",
            ],
            "price": [10.0, 20.0, 30.0, 40.0],
            "freight": [1.0, 2.0, 3.0, 4.0],
            "weight": [100.0] * 4,
            "n_items": [1] * 4,
            "installments": [2.0] * 4,
            "est_days": [10.0] * 4,
            "customer_state": ["SP"] * 4,
            "seller_state": ["RJ"] * 4,
            "category": ["café,livros", "books", "validation-only", "books"],
            "payment_type": ["credit_card"] * 4,
            "late": [0, 1, 0, 1],
        })
        self.frame.to_csv(self.snapshot, index=False, encoding="utf-8")

    def test_valid_csv_preserves_file_and_string_ids(self):
        before = self.snapshot.read_bytes()
        self.assertEqual(validate(self.snapshot), str(self.snapshot))
        self.assertEqual(self.snapshot.read_bytes(), before)

    def test_missing_columns(self):
        for column in REQUIRED_COLUMNS:
            with self.subTest(column=column):
                self.frame.drop(columns=column).to_csv(self.snapshot, index=False)
                with self.assertRaisesRegex(ValueError, column):
                    validate(self.snapshot)

    def test_empty_csv(self):
        for content in ("", ",".join(REQUIRED_COLUMNS) + "\n"):
            with self.subTest(content=content):
                self.snapshot.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    validate(self.snapshot)

    def test_duplicate_header(self):
        content = self.snapshot.read_text(encoding="utf-8")
        self.snapshot.write_text(content.replace("price,freight", "price,price", 1), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "названия столбцов"):
            validate(self.snapshot)

    def test_malformed_csv_rows(self):
        lines = self.snapshot.read_text(encoding="utf-8").splitlines()
        for row in ("extra," + lines[1], lines[1] + ",extra", lines[1].rsplit(",", 1)[0]):
            with self.subTest(row=row):
                self.snapshot.write_text(lines[0] + "\n" + row + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Число полей"):
                    validate(self.snapshot)

    def test_invalid_order_ids(self):
        for value in ("001", "", "   ", None):
            with self.subTest(value=value):
                frame = self.frame.copy()
                frame.loc[1, "order_id"] = value
                frame.to_csv(self.snapshot, index=False)
                with self.assertRaisesRegex(ValueError, "order_id"):
                    validate(self.snapshot)

    def test_invalid_numbers(self):
        for column in [*NUMERIC_COLUMNS, "late"]:
            for value in ("NaN", "inf", "-inf", "text"):
                with self.subTest(column=column, value=value):
                    frame = self.frame.astype({column: str})
                    frame.loc[0, column] = value
                    frame.to_csv(self.snapshot, index=False)
                    with self.assertRaisesRegex(ValueError, column):
                        validate(self.snapshot)

    def test_invalid_labels(self):
        for value in ("-1", "0.5", "2", "True"):
            with self.subTest(value=value):
                frame = self.frame.astype({"late": str})
                frame.loc[0, "late"] = value
                frame.to_csv(self.snapshot, index=False)
                with self.assertRaisesRegex(ValueError, "late"):
                    validate(self.snapshot)

    def test_invalid_dates(self):
        for value in ("", "not-a-date", "NaT", "123"):
            with self.subTest(value=value):
                frame = self.frame.copy()
                frame.loc[0, "order_purchase_timestamp"] = value
                frame.to_csv(self.snapshot, index=False)
                with self.assertRaises(ValueError):
                    validate(self.snapshot)

    def test_validate_has_no_side_effects(self):
        active = False

        def guard(event, args):
            if not active:
                return
            if event == "open":
                write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
                if args[2] & write_flags:
                    raise AssertionError("validate попыталась открыть файл для записи")
            forbidden = ("os.mkdir", "os.chdir", "os.remove", "os.rename", "os.system")
            if event in forbidden or event.startswith(("socket.", "subprocess.", "os.spawn")):
                raise AssertionError(f"Побочное действие validate: {event}")

        sys.addaudithook(guard)
        active = True
        try:
            with patch.dict(os.environ, {}, clear=True):
                validate(self.snapshot)
        finally:
            active = False

    def test_validate_read_only_file_and_directory(self):
        before = self.snapshot.read_bytes()
        self.snapshot.chmod(0o400)
        self.snapshot.parent.chmod(0o500)
        try:
            self.assertEqual(validate(self.snapshot), str(self.snapshot))
            self.assertEqual(self.snapshot.read_bytes(), before)
        finally:
            self.snapshot.parent.chmod(0o700)
            self.snapshot.chmod(0o600)

    def test_validate_reordered_columns_and_quoted_text(self):
        frame = self.frame.copy()
        frame.loc[0, "category"] = 'café, книги\n"новинки"'
        frame["extra_column"] = "ignored"
        frame[frame.columns[::-1]].to_csv(self.snapshot, index=False, encoding="utf-8")
        before = self.snapshot.read_bytes()
        self.assertEqual(validate(self.snapshot), str(self.snapshot))
        self.assertEqual(self.snapshot.read_bytes(), before)

    def test_prepare_boundary_hash_and_repeat(self):
        before = self.snapshot.read_bytes()
        with patch.dict(os.environ, {"OLIST_CUTOFF": "2017-07-31"}):
            prepared = prepare(self.snapshot)
            self.assertEqual(prepare(self.snapshot), prepared)
        self.assertEqual(prepared["snapshot_sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(self.snapshot.read_bytes(), before)
        self.assertEqual((prepared["train_rows"], prepared["validation_rows"]), (2, 2))
        train = pd.read_csv(prepared["train_path"], dtype={"order_id": str})
        validation = pd.read_csv(prepared["validation_path"], dtype={"order_id": str})
        self.assertEqual(train.order_id.tolist(), ["001", "1"])
        self.assertEqual(validation.order_id.tolist(), ["003", "004"])
        self.assertNotIn("validation-only", train.category.tolist())
        self.assertEqual(train.category.iloc[0], "café,livros")

    def test_prepare_rejects_unusable_splits_before_writing(self):
        for cutoff in ("2017-01-01", "2018-01-01", "invalid"):
            with self.subTest(cutoff=cutoff), patch.dict(os.environ, {"OLIST_CUTOFF": cutoff}):
                with self.assertRaises(ValueError):
                    prepare(self.snapshot)
                self.assertEqual(list(self.snapshot.parent.iterdir()), [self.snapshot])

    def test_prepare_requires_both_classes_in_each_split(self):
        self.frame.loc[:1, "late"] = 0
        self.frame.to_csv(self.snapshot, index=False)
        with patch.dict(os.environ, {"OLIST_CUTOFF": "2017-07-31"}):
            with self.assertRaisesRegex(ValueError, "both classes"):
                prepare(self.snapshot)

    def test_train_rejects_changed_snapshot_before_mlflow(self):
        with patch.dict(os.environ, {"OLIST_CUTOFF": "2017-07-31"}):
            prepared = prepare(self.snapshot)
        # Данные те же, но байты снимка изменились: хеш должен это заметить.
        with self.snapshot.open("a", encoding="utf-8") as file:
            file.write("\n")
        with patch("mlflow.set_experiment") as experiment, patch("mlflow.start_run") as run:
            with self.assertRaisesRegex(ValueError, "снимок изменился"):
                train(prepared, dag_run_id="test_changed_snapshot")
            experiment.assert_not_called()
            run.assert_not_called()

    def test_train_rejects_inconsistent_split_counts(self):
        with patch.dict(os.environ, {"OLIST_CUTOFF": "2017-07-31"}):
            prepared = prepare(self.snapshot)
        train_path = Path(prepared["train_path"])
        pd.read_csv(train_path).iloc[:1].to_csv(train_path, index=False)
        with patch("mlflow.set_experiment") as experiment, patch("mlflow.start_run") as run:
            with self.assertRaisesRegex(ValueError, "Размеры выборок"):
                train(prepared, dag_run_id="test_changed_split")
            experiment.assert_not_called()
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
