import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from dl_model_architecture import PanicMeterModel
import random
import torch.nn.functional as F

# Creating a custom dataset
class FloodPanicDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform = None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Keeping only the images which are assigned a panic meter.
        self.data = self.data[self.data["img_name"].apply(
            lambda x: os.path.exists(os.path.join(img_dir, x))
        )].reset_index(drop = True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        panic_value_str = self.data.iloc[idx, 1]
        if panic_value_str == "High":
            panic_value_int = 3
        elif panic_value_str == "Medium":
            panic_value_int = 2
        elif panic_value_str == "Low":
            panic_value_int = 1
        # else:
        #     panic_value_int = 0  # For images which can't be classified as a calamity or disaster.

        panic_value = torch.tensor(panic_value_str, dtype = torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, panic_value

# Preparing the dataset
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness = 0.2, contrast = 0.2),
    transforms.TrivialAugmentWide(num_magnitude_bins = 20),
    transforms.AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = FloodPanicDataset(csv_file="data/images_with_panic_meter.csv", img_dir="data/flood", transform=train_transform)

print("Total images in dataset : ", len(dataset))

train_idx, val_idx = train_test_split(range(len(dataset)), test_size = 0.2, random_state = 21)
train_subset = torch.utils.data.Subset(dataset, train_idx)
val_subset = torch.utils.data.Subset(dataset, val_idx)

BATCH_SIZE = 64
train_loader = DataLoader(train_subset, batch_size = BATCH_SIZE, shuffle = True, pin_memory = True, persistent_workers = True, num_workers = 4)
val_loader = DataLoader(val_subset, batch_size = BATCH_SIZE, shuffle = False, pin_memory = True, persistent_workers = True, num_workers = 4)

# Training the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load pretrained MobileNetV2
model = models.mobilenet_v2(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# MobileNetV2 final layer is inside model.classifier
# Replace it with a regression head
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

# Only the new layer's parameters will be trainable
for param in model.classifier[1].parameters():
    param.requires_grad = True
# model = PanicMeterModel(img_size = 224)
model = model.to(device)
# state_dict = torch.load("models/panic_level_custom_model.pth")
# model.load_state_dict(state_dict)

# Focal Loss loss func

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction="mean"):
        """
        alpha: float, list, or tensor of shape (num_classes,)
        gamma: focusing parameter
        """
        super(FocalLoss, self).__init__()
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha  # can be scalar or None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: (batch_size, num_classes) logits
        targets: (batch_size,) ground-truth class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")  # per-sample CE
        pt = torch.exp(-ce_loss)  # probability of true class

        # Handle alpha
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                # move to same device
                self.alpha = self.alpha.to(inputs.device)
                at = self.alpha[targets]  # pick alpha for each sample's class
            else:
                at = self.alpha
        else:
            at = 1.0

        focal_loss = at * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

alpha = torch.tensor([0.35, 0.25, 0.40, 0.45])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 1e-4)

num_epochs = 300


def train():
    for epoch in tqdm(range(num_epochs)):
        # Training
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            optimizer.zero_grad()
            outputs = model(imgs).squeeze()  # force shape [B]
            labels = labels.squeeze()  # force shape [B]

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).float()

                outputs = model(imgs).squeeze()
                labels = labels.squeeze()

                loss = criterion(outputs, labels)
                val_loss += loss.item()

        if epoch % 20 == 0 or epoch + 1 == num_epochs:
            print(f"Epoch [{epoch + 1}/{num_epochs}] "
                  f"Train loss: {train_loss / len(train_loader):.4f} | RMSE : {(train_loss / len(train_loader))**(1/2)} "
                  f"Val loss: {val_loss / len(val_loader):.4f}  | RMSE : {(val_loss / len(val_loader))**(1/2)}")

    # correct = 0
    # total = 0
    # model.eval()
    # with torch.no_grad():
    #     for x, y in val_loader:
    #         x, y = x.to(device), y.to(device)
    #         outputs = model(x)  # logits
    #         _, preds = torch.max(outputs, 1)  # predicted class
    #         correct += (preds == y).sum().item()
    #         total += y.size(0)

    # val_accuracy = correct / total * 100
    # print(f"Validation Accuracy: {val_accuracy:.2f}%")

    # Saving the model
    torch.save(model.state_dict(), "models/panic_meter_mobilenet_model.pth")
    print("Model saved successfully.")

    # Current best model : "models/panic_level_custom_model_V3.pth"

if __name__ == "__main__":
    train()