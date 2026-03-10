import pandas as pd
import pytest

from src.config import ExperimentConfig, Paths
from src.split import stratified_split


def _balanced_df(n: int = 100):
    return pd.DataFrame(
        {
            "feature": list(range(n)),
            "target": [0] * (n // 2) + [1] * (n // 2),
        }
    )


class TestStratifiedSplit:
    def test_split_sizes_match_ratio(self):
        df = _balanced_df(100)
        train, test = stratified_split(df, "target", test_size=0.2, seed=42)
        assert len(train) == 80
        assert len(test) == 20

    def test_no_row_overlap(self):
        df = _balanced_df(100)
        train, test = stratified_split(df, "target", test_size=0.2, seed=42)
        overlap = set(train.index) & set(test.index)
        assert len(overlap) == 0

    def test_all_rows_preserved(self):
        df = _balanced_df(100)
        train, test = stratified_split(df, "target", test_size=0.2, seed=42)
        assert len(train) + len(test) == len(df)

    def test_stratification_preserves_class_ratio(self):
        df = _balanced_df(100)
        train, test = stratified_split(df, "target", test_size=0.2, seed=42)
        train_ratio = train["target"].mean()
        test_ratio = test["target"].mean()
        assert train_ratio == pytest.approx(0.5, abs=0.05)
        assert test_ratio == pytest.approx(0.5, abs=0.05)

    def test_seed_determinism(self):
        df = _balanced_df(100)
        train1, test1 = stratified_split(df, "target", test_size=0.2, seed=42)
        train2, test2 = stratified_split(df, "target", test_size=0.2, seed=42)
        pd.testing.assert_frame_equal(
            train1.reset_index(drop=True), train2.reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            test1.reset_index(drop=True), test2.reset_index(drop=True)
        )

    def test_different_seed_gives_different_split(self):
        df = _balanced_df(100)
        _, test1 = stratified_split(df, "target", test_size=0.2, seed=42)
        _, test2 = stratified_split(df, "target", test_size=0.2, seed=99)
        assert not test1.index.equals(test2.index)


class TestSplitCLI:
    """Tests for the split CLI command (persisted train/cal/test)."""

    def test_cmd_split_creates_all_files(self, tmp_path):
        """split command should produce train.parquet, cal.parquet, test.parquet + metadata."""
        import json

        # Setup: create a minimal dataset.parquet
        n = 200
        df = pd.DataFrame(
            {
                "lead_time": list(range(n)),
                "is_canceled": [0] * (n // 2) + [1] * (n // 2),
            }
        )
        processed = tmp_path / "data" / "processed"
        processed.mkdir(parents=True)
        df.to_parquet(processed / "dataset.parquet", index=False)

        paths = Paths.__new__(Paths)
        object.__setattr__(paths, "project_root", tmp_path)
        object.__setattr__(paths, "data_dir", tmp_path / "data")
        object.__setattr__(paths, "data_raw", tmp_path / "data" / "raw")
        object.__setattr__(paths, "data_processed", processed)
        object.__setattr__(paths, "models", tmp_path / "models")
        object.__setattr__(paths, "reports", tmp_path / "reports")
        object.__setattr__(paths, "reports_metrics", tmp_path / "reports" / "metrics")
        object.__setattr__(
            paths, "reports_predictions", tmp_path / "reports" / "predictions"
        )
        object.__setattr__(
            paths, "reports_monitoring", tmp_path / "reports" / "monitoring"
        )

        cfg = ExperimentConfig(target_col="is_canceled")

        from src.cli.split import cmd_split

        cmd_split(paths, cfg)

        # All three split files must exist
        assert (processed / "train.parquet").exists()
        assert (processed / "cal.parquet").exists()
        assert (processed / "test.parquet").exists()

        # Metadata must exist
        meta_path = tmp_path / "reports" / "metrics" / "split_metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["seed"] == 42
        assert "splits" in meta

        # Row counts must sum to original
        train_df = pd.read_parquet(processed / "train.parquet")
        cal_df = pd.read_parquet(processed / "cal.parquet")
        test_df = pd.read_parquet(processed / "test.parquet")
        assert len(train_df) + len(cal_df) + len(test_df) == n

    def test_no_row_overlap_across_splits(self, tmp_path):
        """No single row should appear in more than one split."""
        n = 200
        df = pd.DataFrame(
            {
                "row_id": list(range(n)),
                "is_canceled": [0] * (n // 2) + [1] * (n // 2),
            }
        )
        processed = tmp_path / "data" / "processed"
        processed.mkdir(parents=True)
        df.to_parquet(processed / "dataset.parquet", index=False)

        paths = Paths.__new__(Paths)
        object.__setattr__(paths, "project_root", tmp_path)
        object.__setattr__(paths, "data_dir", tmp_path / "data")
        object.__setattr__(paths, "data_raw", tmp_path / "data" / "raw")
        object.__setattr__(paths, "data_processed", processed)
        object.__setattr__(paths, "models", tmp_path / "models")
        object.__setattr__(paths, "reports", tmp_path / "reports")
        object.__setattr__(paths, "reports_metrics", tmp_path / "reports" / "metrics")
        object.__setattr__(
            paths, "reports_predictions", tmp_path / "reports" / "predictions"
        )
        object.__setattr__(
            paths, "reports_monitoring", tmp_path / "reports" / "monitoring"
        )

        cfg = ExperimentConfig(target_col="is_canceled")

        from src.cli.split import cmd_split

        cmd_split(paths, cfg)

        train_ids = set(pd.read_parquet(processed / "train.parquet")["row_id"])
        cal_ids = set(pd.read_parquet(processed / "cal.parquet")["row_id"])
        test_ids = set(pd.read_parquet(processed / "test.parquet")["row_id"])

        assert len(train_ids & cal_ids) == 0, "train ∩ cal overlap"
        assert len(train_ids & test_ids) == 0, "train ∩ test overlap"
        assert len(cal_ids & test_ids) == 0, "cal ∩ test overlap"
        assert len(train_ids | cal_ids | test_ids) == n

    def test_stratification_preserved_in_all_splits(self, tmp_path):
        """Positive class ratio should be roughly equal across all splits."""
        n = 1000
        pos_rate = 0.37
        n_pos = int(n * pos_rate)
        df = pd.DataFrame(
            {
                "feature": list(range(n)),
                "is_canceled": [0] * (n - n_pos) + [1] * n_pos,
            }
        )
        processed = tmp_path / "data" / "processed"
        processed.mkdir(parents=True)
        df.to_parquet(processed / "dataset.parquet", index=False)

        paths = Paths.__new__(Paths)
        object.__setattr__(paths, "project_root", tmp_path)
        object.__setattr__(paths, "data_dir", tmp_path / "data")
        object.__setattr__(paths, "data_raw", tmp_path / "data" / "raw")
        object.__setattr__(paths, "data_processed", processed)
        object.__setattr__(paths, "models", tmp_path / "models")
        object.__setattr__(paths, "reports", tmp_path / "reports")
        object.__setattr__(paths, "reports_metrics", tmp_path / "reports" / "metrics")
        object.__setattr__(
            paths, "reports_predictions", tmp_path / "reports" / "predictions"
        )
        object.__setattr__(
            paths, "reports_monitoring", tmp_path / "reports" / "monitoring"
        )

        cfg = ExperimentConfig(target_col="is_canceled")

        from src.cli.split import cmd_split

        cmd_split(paths, cfg)

        for fname in ("train.parquet", "cal.parquet", "test.parquet"):
            split_df = pd.read_parquet(processed / fname)
            split_rate = split_df["is_canceled"].mean()
            assert split_rate == pytest.approx(
                pos_rate, abs=0.05
            ), f"{fname}: positive_rate={split_rate:.3f} expected≈{pos_rate}"
