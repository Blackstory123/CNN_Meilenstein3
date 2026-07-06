import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Dataset

from studentv1 import StudentCNN

"""
Knowledge-Distillation-Experiment mit einem 32x32-Teacher.

Dieses Script untersucht, ob ein Teacher mit derselben Eingabeauflösung
wie der Student bessere Soft Targets liefert.

Im Ergebnis zeigte sich, dass der 32x32-Teacher schlechter abschnitt
als der stärkere ResNet18-Teacher mit 224x224-Eingaben.
"""

SCRIPT_DIR = Path(__file__).resolve().parent
MILESTONE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = MILESTONE_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = MILESTONE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

STUDENT_KD_PATH = MODEL_DIR / "best_student_kdv4_32.pth"

TEACHER_PATH = PROJECT_DIR / "models" / "Station2" / "best_resnet18_v4_32.pth"

class DistillationDataset(Dataset):
    def __init__(self, dataset, student_transform, teacher_transform):
        self.dataset = dataset
        self.student_transform = student_transform
        self.teacher_transform = teacher_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        student_image = self.student_transform(image)
        teacher_image = self.teacher_transform(image)

        return student_image, teacher_image, label

def create_teacher_model(num_classes=10):
    model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

def distillation_loss(
    student_outputs,
    teacher_outputs,
    labels,
    temperature=4.0,
    alpha=0.7
):
    ce_loss = F.cross_entropy(student_outputs, labels)

    kd_loss = F.kl_div(
        F.log_softmax(student_outputs / temperature, dim=1),
        F.softmax(teacher_outputs / temperature, dim=1),
        reduction="batchmean"
    ) * (temperature ** 2)

    loss = alpha * kd_loss + (1 - alpha) * ce_loss

    return loss, ce_loss, kd_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

student_transform = transforms.Compose([
    transforms.ToTensor()
])

teacher_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

full_train_dataset = datasets.CIFAR10(
    root=str(DATA_DIR),
    train=True,
    download=False,
    transform=None
)

train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

generator = torch.Generator().manual_seed(42)

train_raw, val_raw = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=generator
)

train_dataset = DistillationDataset(
    train_raw,
    student_transform,
    teacher_transform
)

val_dataset = DistillationDataset(
    val_raw,
    student_transform,
    teacher_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)


test_dataset = datasets.CIFAR10(
    root=str(DATA_DIR),
    train=False,
    download=False,
    transform=student_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

student = StudentCNN(num_classes=10)
student = student.to(device)

teacher = create_teacher_model(num_classes=10)

if not TEACHER_PATH.exists():
    raise FileNotFoundError(f"Teacher-Modell nicht gefunden: {TEACHER_PATH}")

teacher.load_state_dict(
    torch.load(
        TEACHER_PATH,
        map_location=device
    )
)

teacher = teacher.to(device)
teacher.eval()

for param in teacher.parameters():
    param.requires_grad = False

student_params = sum(p.numel() for p in student.parameters())
teacher_params = sum(p.numel() for p in teacher.parameters())

print(f"Student Parameter: {student_params:,}")
print(f"Teacher Parameter: {teacher_params:,}")

optimizer = optim.Adam(
    student.parameters(),
    lr=0.001
)

epochs = 10
best_val_accuracy = 0.0

#Base KD-Varianten
#temperature = 4.0
#alpha = 0.7

#V2 KD-Varianten
#temperature = 4.0
#alpha = 0.5

#V3 KD-Varianten
#temperature = 2.0  
#alpha = 0.7

#V4 KD-Varianten
temperature = 6.0
alpha = 0.7

#V5 KD-Varianten
#temperature = 4.0
#alpha = 0.9

for epoch in range(epochs):
    student.train()

    running_loss = 0.0
    running_ce_loss = 0.0
    running_kd_loss = 0.0

    for batch_idx, (student_images, teacher_images, labels) in enumerate(train_loader):
        student_images = student_images.to(device)
        teacher_images = teacher_images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        student_outputs = student(student_images)

        with torch.no_grad():
            teacher_outputs = teacher(teacher_images)

        loss, ce_loss, kd_loss = distillation_loss(
            student_outputs,
            teacher_outputs,
            labels,
            temperature=temperature,
            alpha=alpha
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_ce_loss += ce_loss.item()
        running_kd_loss += kd_loss.item()

        if batch_idx % 100 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"Batch {batch_idx}/{len(train_loader)}"
            )

    train_loss = running_loss / len(train_loader)
    avg_ce_loss = running_ce_loss / len(train_loader)
    avg_kd_loss = running_kd_loss / len(train_loader)

    student.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for student_images, teacher_images, labels in val_loader:
            student_images = student_images.to(device)
            labels = labels.to(device)

            outputs = student(student_images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Loss: {train_loss:.4f}, "
        f"CE Loss: {avg_ce_loss:.4f}, "
        f"KD Loss: {avg_kd_loss:.4f}, "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(student.state_dict(), STUDENT_KD_PATH)
        print(f"Bestes KD-Student-Modell gespeichert: {STUDENT_KD_PATH}")

student.load_state_dict(
    torch.load(
        STUDENT_KD_PATH,
        map_location=device
    )
)

student.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = student(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total

print(f"Beste Validation Accuracy: {best_val_accuracy:.2f}%")
print(f"Finale Test Accuracy KD-Student: {test_accuracy:.2f}%")
print(f"Gespeichertes Modell: {STUDENT_KD_PATH}")
