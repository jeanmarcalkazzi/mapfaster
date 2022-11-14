import ast
from pathlib import Path

import pandas as pd


def get_train_val_test_df(
    maps_analytics_fpath: Path,
    train_ratio: float,
    val_ratio: float,
    seed=None,
    sample=None,
):
    df: pd.DataFrame = pd.read_csv(maps_analytics_fpath)

    # Used during development to work on a subset of the data
    if sample is not None:
        df = df.sample(frac=sample, random_state=seed)

    # Values are saved as string representation of a list
    # Convert to list
    df["y2"] = df["y2"].apply(ast.literal_eval)
    df["y3"] = df["y3"].apply(ast.literal_eval)

    df_trainval = df.sample(frac=train_ratio + val_ratio, random_state=seed)
    df_test = df.drop(df_trainval.index)
    df_train = df_trainval.sample(
        frac=train_ratio / (train_ratio + val_ratio), random_state=seed
    )
    df_val = df_trainval.drop(df_train.index)

    return df_train, df_val, df_test
