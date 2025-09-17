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
            panic_value_int = 2
        elif panic_value_str == "Low":
            panic_value_int = 1
        else:
            panic_value_int = 0
        panic_value = torch.tensor(panic_value_int, dtype = torch.long)

        if self.transform:
            image = self.transform(image)

        return image, panic_value

# Preparing the dataset
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness = 0.2, contrast = 0.2),
    transforms.TrivialAugmentWide(num_magnitude_bins = 25),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = FloodPanicDataset(csv_file="data/img_panic_level.csv", img_dir="data/flood", transform=train_transform)

print("Total images in dataset : ", len(dataset))

train_idx, val_idx = train_test_split(range(len(dataset)), test_size = 0.2, random_state = 21)
train_subset = torch.utils.data.Subset(dataset, train_idx)
val_subset = torch.utils.data.Subset(dataset, val_idx)

BATCH_SIZE = 64
train_loader = DataLoader(train_subset, batch_size = BATCH_SIZE, shuffle = True, pin_memory = True, persistent_workers = True, num_workers = 6)
val_loader = DataLoader(val_subset, batch_size = BATCH_SIZE, shuffle = False, pin_memory = True, persistent_workers = True, num_workers = 6)

# Training the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = models.resnet18(pretrained = True)
model.fc = nn.Linear(model.fc.in_features, 3)   # regression head
# model = PanicMeterModel(img_size = 256)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 1e-4)

num_epochs = 200

def train():
    for epoch in tqdm(range(num_epochs)):
        # Training
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device, non_blocking = True), labels.to(device, non_blocking = True)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device, non_blocking = True), labels.to(device, non_blocking = True)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        if epoch % 20 == 0 or epoch + 1 == num_epochs:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train loss: {train_loss/len(train_loader):.4f} "
                  f"Val loss: {val_loss/len(val_loader):.4f}")

    # Saving the model
    torch.save(model.state_dict(), "models/panic_level_custom_model.pth")
    print("Model saved successfully.")

if __name__ == "__main__":
    train()