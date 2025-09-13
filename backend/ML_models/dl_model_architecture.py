import torch
import torch.nn as nn

class PanicMeterModel(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Computing feature map size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            out = self.features(dummy)
            n_features = out.numel()  # total elements for one sample

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = n_features, out_features = 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_features = 256, out_features = 64),
            nn.ReLU(),
            nn.Linear(in_features = 64, out_features = 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x
