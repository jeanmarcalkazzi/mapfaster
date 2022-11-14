import hashlib
from pathlib import Path

from src.dataloader import get_train_val_test_df


def test_data_split_is_deterministic_with_same_seed():
    df_train, df_val, df_test = get_train_val_test_df(
        maps_analytics_fpath=Path("dataset/dataframes/maps_analytics.csv"),
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42,
    )
    assert (
        hashlib.md5(str(df_train).encode("utf-8")).hexdigest()
        == "84387e4e4b9f38fd3ff60920bae2998b"
    )
    assert (
        hashlib.md5(str(df_val).encode("utf-8")).hexdigest()
        == "8ca27037645d5985e3b6ae104d6cc6d7"
    )
    assert (
        hashlib.md5(str(df_test).encode("utf-8")).hexdigest()
        == "6c4b19232ee4cd9f2f91eb0bb62aec0a"
    )
