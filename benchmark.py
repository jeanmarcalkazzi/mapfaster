import os
from functools import partial

import numpy as np
import pandas as pd
import torch
from fastai.callback.tracker import CSVLogger, SaveModelCallback
from fastai.callback.wandb import WandbCallback
from fastai.vision.all import (Adam, CategoryBlock, ColReader, DataBlock,
                               DataLoaders, EarlyStoppingCallback, ImageBlock,
                               Learner, Normalize, RandomSplitter,
                               RegressionBlock, imagenet_stats)

import wandb
from src.dataloader import get_train_val_test_df
from src.loss import MAPFASTERLoss
from src.metrics import (SumMetric, mapfast_accuracy,
                         mapfast_accuracy_per_maptype, mapfast_coverage,
                         mapfast_score)
from src.model.mapfaster import MAPFASTER
from src.transform import get_item_tfms
from src.utils import fix_random_seed

torch.cuda.set_device(0)

rng = np.random.RandomState(seed=12345)
seeds = np.arange(10**5)
rng.shuffle(seeds)
# Set it to the number of runs you want.
iterations = int(os.environ.get("iterations", 50))
seeds = seeds[:iterations]

for seed in seeds:

    run = wandb.init(
        entity="YOUR_ENTITY",
        project="YOUR_PROJECT",
        job_type="train",
        tags=["YOUR_TAGS"],
        reinit=True,
    )

    config = run.config
    config = {
        "batch_size": 64,
        "epochs": 9,
        "opt_lr": 0.004,  # Irrelevant with the lr_max set already [if fit_one_cycle].
        "one_cycle.lr_max": 0.001,
        "encoder_se_reduction": 2,
        "encoder_body_channels": 4,
        "encoder_head_channels": 2,
        "encoder_dropout": 0,
        "block0_se_reduction": 4,
        "block0_body_in_channels": 8,
        "block0_body_out_channels": 16,
        "block0_head_channels": 8,
        "block0_dropout": 0,
        "predictor0_dropout": 0,
        "predictor0_linear_channels": 512,
        "seed": seed,
    }

    SEED = os.environ.get("TRAINING_SEED", None)
    FIX_SEED = os.environ.get("FIX_SEED", False)

    SEED = int(SEED) if SEED is not None else None

    if "seed" in config:
        print(f"Using seed {config['seed']} from run config")
        SEED = config["seed"]
        FIX_SEED = True

    print(SEED, FIX_SEED)

    if FIX_SEED and SEED:
        fix_random_seed(SEED, cuda=True)
        print(f"Fixed seed to {SEED}")

    image_size = (320, 320)
    train_ratio = 0.8
    val_ratio = 0.1

    df_train, df_val, df_test = get_train_val_test_df(
        maps_analytics_fpath="dataset/dataframes/maps_analytics.csv",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=SEED,
    )

    dblock_train_val: DataBlock = DataBlock(
        blocks=(
            ImageBlock,  # Input Image
            CategoryBlock,  # Class
            RegressionBlock,  # Fin
            RegressionBlock,  # Pair
            CategoryBlock,  # Map Class
            RegressionBlock,  # Runtime of BCP
            RegressionBlock,  # Runtime of CBS
            RegressionBlock,  # Runtime of CBSH
            RegressionBlock,  # Runtime of SAT
        ),
        n_inp=1,
        get_x=ColReader("map_name", pref="dataset/images/", suff=".png"),
        get_y=[
            ColReader("y1"),
            ColReader("y2"),
            ColReader("y3"),
            ColReader("map_class"),
            ColReader("bcp_runtime"),
            ColReader("cbs_runtime"),
            ColReader("cbsh_runtime"),
            ColReader("sat_runtime"),
        ],
        splitter=RandomSplitter(
            valid_pct=1 - (train_ratio / (train_ratio + val_ratio)),
            seed=SEED,
        ),
        item_tfms=get_item_tfms(im_size=image_size),
        batch_tfms=Normalize.from_stats(*imagenet_stats),
    )

    dls: DataLoaders = dblock_train_val.dataloaders(
        pd.concat([df_train, df_val]),
        shuffle=True,
        bs=config["batch_size"],
        drop_last=True,
    )

    learner: Learner = Learner(
        dls=dls,
        model=MAPFASTER.init_from_config(config),
        loss_func=MAPFASTERLoss(
            aux_task_fin_activated=False,
            aux_task_pair_activated=False,
        ),
        opt_func=partial(Adam, lr=config["opt_lr"]),
        metrics=[mapfast_accuracy, mapfast_coverage, SumMetric(mapfast_score)],
        cbs=[
            # I believe that the task is limited by the dataset, so will be any architecture.
            EarlyStoppingCallback(monitor="valid_loss", min_delta=0.01, patience=2),
            WandbCallback(
                log="all",
                log_model=True,
                log_preds=False,
                log_dataset="dataset/dataframes",
                seed=SEED,
            ),
            SaveModelCallback(every_epoch=True, fname=f"{run.sweep_id}_{run.name}"),
            CSVLogger(fname=f"history/{run.sweep_id}_{run.name}.csv"),
        ],
    )

    dls_test: DataLoaders = learner.dls.test_dl(df_test, with_labels=True)

    # Train Model
    learner.fit_one_cycle(
        n_epoch=config["epochs"],
        lr_max=config["one_cycle.lr_max"],
    )

    # Evaluate Model
    with torch.inference_mode():
        test_pred, test_targ = learner.get_preds(dl=dls_test)
        test_mapfast_accuracy = mapfast_accuracy(test_pred, *test_targ, axis=-1)
        test_mapfast_coverage = mapfast_coverage(test_pred, *test_targ, axis=-1)
        test_mapfast_score = mapfast_score(
            test_pred, *[tt.cuda() for tt in test_targ], axis=-1
        )
        test_mapfast_accuracy_per_map = mapfast_accuracy_per_maptype(
            test_pred, *test_targ, vocab=dls_test.vocab[1], axis=-1
        )
        print(f"Test MAPFASTER Accuracy: {test_mapfast_accuracy:.2f}")
        wandb.log({"test/accuracy": test_mapfast_accuracy})
        print(f"Test MAPFASTER Coverage: {test_mapfast_coverage:.2f}")
        wandb.log({"test/coverage": test_mapfast_coverage})
        print(f"Test MAPFASTER Score: {test_mapfast_score:.2f}")
        wandb.log({"test/score": test_mapfast_score})
        print(f"Test MAPFASTER Accuracy per MapType: {test_mapfast_accuracy_per_map}")
        for map_name in test_mapfast_accuracy_per_map:
            wandb.log(
                {
                    "test/accuracy_per_map/"
                    + map_name: test_mapfast_accuracy_per_map[map_name]
                }
            )

    run.finish()
