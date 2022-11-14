# MAPFASTER


**Paper Abstract**

> Portfolio-based algorithm selection can help in choosing the best suited algorithm for a given task while leveraging the complementary strengths of the candidates. Solving the Multi-Agent Path Finding (MAPF) problem optimally has been proven to be NP-Hard. Furthermore, no single optimal algorithm has been shown to have the fastest runtime for all MAPF problem instances, and there are no proven approaches for when to use each algorithm. To address these challenges, we develop MAPFASTER, a smaller and more accurate deep learning based architecture aiming to be deployed in fleet management systems to select the fastest MAPF solver in a multi-robot setting. MAPF problem instances are encoded as images and passed to the model for classification into one of the portfolio's candidates. We evaluate our model against state-of-the-art Optimal-MAPF-Algorithm selectors, showing `+5.42%` improvement in accuracy while being `7.1times` faster to train. The dataset, code and analysis used in this research can be found at https://github.com/jeanmarcalkazzi/mapfaster.


## Description

This repo holds the code for `MAPFASTER`.  
It is intended to be simple to read and modify, while having a `wandb` integration for better monitoring of your training.  Please open a PR if you believe there is anything which could improve the use of this repo.

Should you focus on finding the best-state-of-the-art-architecture-while-having-transformers-and-attention to beat this?  Please don't.  The reason is, the dataset does not show enough features for a bigger model to be relevant in such a scenario.  Look at the imbalance in the classification and the dataset format if you care about improving the MAPF Algorithm Selection aspect.  If you care about publishing yet another paper, I can't stop you anyways, so be my guest and improve 0.1% on this work.

## Just tell me how to use it

Fine, here you go.  

First, make sure you have `git-lfs` [installed](https://github.com/git-lfs/git-lfs/wiki/Installation), before cloning this repo.

This repo has a `Makefile` to make your life easier.  
You just need to have `python3.9` installed on your system to be used by `poetry`.  
To run a benchmark you can just use `make iterations=40 benchmark-local` to run 40 different train/val/test splits with 40 different random seeds.  To include `wandb` in your benchmark, you need to fill in relevant info in the `wandb.init` function inside `benchmark.py` and run `make iterations=40 benchmark`.
