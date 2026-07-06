import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from studentv1 import StudentCNN


SCRIPT_DIR = Path(__file__).resolve().parent
MILESTONE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = MILESTONE_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = MILESTONE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_student_baseline.pth"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


transform = transforms.Compose([
    transforms.ToTensor()
])


full_train_dataset = datasets.CIFAR10(
    root=str(DATA_DIR),
    train=True,
    download=False,
    transform=transform
)

train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)


test_dataset = datasets.CIFAR10(
    root=str(DATA_DIR),
    train=False,
    download=False,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


model = StudentCNN(num_classes=10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 10
best_val_accuracy = 0.0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Loss: {train_loss:.4f}, "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Bestes Student-Modell gespeichert: {MODEL_PATH}")


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total

print(f"Finale Test Accuracy: {test_accuracy:.2f}%")
print(f"Gespeichertes Modell: {MODEL_PATH}")