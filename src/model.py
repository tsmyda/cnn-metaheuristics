import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class TunableCNN(nn.Module):
    def __init__(
        self,
        image_channels: int,
        image_size: int,
        num_classes: int,
        num_blocks: int,
        filters_1: int,
        filters_2: int,
        filters_3: int,
        kernel_size: int,
        dropout: float,
        dense_units: int,
    ):
        super().__init__()

        filters = [filters_1, filters_2, filters_3][:num_blocks]
        padding = kernel_size // 2

        layers = []
        in_channels = image_channels
        current_size = image_size

        for out_channels in filters:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2))
            layers.append(nn.Dropout(dropout))

            in_channels = out_channels
            current_size //= 2

        self.features = nn.Sequential(*layers)

        flattened_dim = in_channels * current_size * current_size

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x