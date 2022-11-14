from functools import wraps

import torch.nn as nn
from fastai.losses import CrossEntropyLossFlat, FocalLoss


class MAPFASTERLoss(nn.Module):
    def __init__(self, aux_task_fin_activated=False, aux_task_pair_activated=False):
        super().__init__()

        self.classification_loss = nn.CrossEntropyLoss()

        self.criterion2 = nn.BCEWithLogitsLoss()
        self.criterion3 = nn.BCEWithLogitsLoss()

        self.aux_task_fin_activated = aux_task_fin_activated
        self.aux_task_pair_activated = aux_task_pair_activated

    def forward(self, inp, class_out, fin_out, pair_out, *not_relevant):
        loss = self.classification_loss(inp, class_out)

        if self.aux_task_fin_activated:
            loss += self.criterion2(inp[1], fin_out)

        if self.aux_task_pair_activated:
            if self.aux_task_fin_activated:
                loss += self.criterion3(inp[2], pair_out)
            else:
                loss += self.criterion2(inp[1], pair_out)

        return loss
