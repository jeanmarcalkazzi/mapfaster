import torch.nn as nn

from src.model.modules import SEModule


class MAPFASTER(nn.Module):
    def __init__(
        self,
        encoder_se_reduction=None,
        encoder_body_channels=8,
        encoder_head_channels=8,
        encoder_dropout=None,
        block0_se_reduction=None,
        block0_dropout=None,
        block0_body_in_channels=16,
        block0_body_out_channels=16,
        block0_head_channels=3,
        predictor0_dropout=None,
        predictor0_linear_channels=128,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, encoder_body_channels, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(encoder_body_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                encoder_body_channels,
                encoder_head_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(encoder_head_channels),
            nn.ReLU(inplace=True),
        )

        # if encoder_se_reduction is not None:
        #     self.encoder.add_module(
        #         name="se_block",
        #         module=SEModule(encoder_head_channels, reduction=encoder_se_reduction),
        #     )

        if encoder_dropout is not None:
            self.encoder.add_module(
                name="dropout_block",
                module=nn.Dropout(p=encoder_dropout),
            )

        self.block0 = nn.Sequential(
            nn.Conv2d(
                encoder_head_channels,
                block0_body_in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(block0_body_in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                block0_body_in_channels,
                block0_body_out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(block0_body_out_channels),
            nn.ReLU(inplace=True),
        )

        # if block0_se_reduction is not None:
        #     self.block0.add_module(
        #         name="se_block",
        #         module=SEModule(block0_body_out_channels, reduction=block0_se_reduction),
        #     )

        if block0_dropout is not None:
            self.block0.add_module(
                name="dropout_block",
                module=nn.Dropout(p=block0_dropout),
            )

        self.block0_head = nn.Sequential(
            nn.Conv2d(
                block0_body_out_channels,
                block0_head_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(block0_head_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.predictor0 = nn.Sequential(
            nn.Conv2d(block0_head_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(
                p=0.0 if predictor0_dropout is None else predictor0_dropout,
                inplace=True,
            ),
            nn.Flatten(),
            nn.Linear(
                in_features=1600,  # 1600 for 320x320, 784 for 224x224
                out_features=predictor0_linear_channels,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=predictor0_linear_channels,
                out_features=4,
            ),
        )

        self.apply(self.init_weights)

    @classmethod
    def init_from_config(cls, config):
        return cls(
            encoder_se_reduction=config["encoder_se_reduction"],
            encoder_dropout=config["encoder_dropout"],
            encoder_body_channels=config["encoder_body_channels"],
            encoder_head_channels=config["encoder_head_channels"],
            block0_se_reduction=config["block0_se_reduction"],
            block0_dropout=config["block0_dropout"],
            block0_body_in_channels=config["block0_body_in_channels"],
            block0_body_out_channels=config["block0_body_out_channels"],
            block0_head_channels=config["block0_head_channels"],
            predictor0_dropout=config["predictor0_dropout"],
            predictor0_linear_channels=config["predictor0_linear_channels"],
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.block0(x)
        x = self.block0_head(x)
        x = self.predictor0(x)

        return x

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
