import time
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from studentv1 import StudentCNN

"""
Speed-Test für den Vergleich der Modelle aus Meilenstein 1, 2 und 3.

Verglichen werden:
- StudentKD20 aus Meilenstein 3
- CNN V10 aus Meilenstein 1
- ResNet18 Teacher aus Meilenstein 2

Gemessen werden:
- Parameteranzahl
- geschätzter Speicherbedarf
- Inferenzzeit pro Bild
- Bilder pro Sekunde

Die Messung dient nur dem relativen Vergleich auf demselben System.
"""

SCRIPT_DIR = Path(__file__).resolve().parent
MILESTONE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = MILESTONE_DIR.parent

sys.path.append(str(PROJECT_DIR))

from modelv10 import SimpleCNN

STUDENT_PATH = MILESTONE_DIR / "models" / "best_student_kdv4_doubleepochs.pth"
TEACHER_PATH = PROJECT_DIR / "models" / "Station2" / "best_resnet18_hflip.pth"
CNN_M1_PATH = PROJECT_DIR / "models" / "best_model_v10.pth"


def create_resnet18_teacher(num_classes=10):
    model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def load_weights(model, path, device):
    if not path.exists():
        raise FileNotFoundError(f"Modelldatei nicht gefunden: {path}")

    checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def parameter_size_mb(model):
    return count_parameters(model) * 4 / (1024 ** 2)

# Führt mehrere Forward-Passes mit Dummy-Daten aus.
# Die ersten Durchläufe dienen als Warm-up und werden nicht gemessen.

def benchmark_model(
    model,
    input_size,
    device,
    batch_size=32,
    warmup_runs=5,
    benchmark_runs=50
):
    model.to(device)
    model.eval()

    dummy_input = torch.randn(batch_size, *input_size).to(device)

    with torch.inference_mode():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.inference_mode():
        for _ in range(benchmark_runs):
            _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    total_time = end_time - start_time
    total_images = batch_size * benchmark_runs

    ms_per_image = (total_time / total_images) * 1000
    fps = total_images / total_time

    return ms_per_image, fps


def print_result(name, model, input_size, accuracy, device):
    params = count_parameters(model)
    size_mb = parameter_size_mb(model)

    ms_per_image, fps = benchmark_model(
        model=model,
        input_size=input_size,
        device=device,
        batch_size=32,
        warmup_runs=5,
        benchmark_runs=50
    )

    print("-" * 70)
    print(f"Modell: {name}")
    print(f"Parameter: {params:,}")
    print(f"Speicherbedarf Parameter: {size_mb:.2f} MB")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Eingabegröße: {input_size}")
    print(f"Inferenzzeit pro Bild: {ms_per_image:.4f} ms")
    print(f"Bilder pro Sekunde: {fps:.2f} FPS")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("Lade Modelle...")

student = StudentCNN(num_classes=10)
student = load_weights(student, STUDENT_PATH, device)

cnn_m1 = SimpleCNN()
cnn_m1 = load_weights(cnn_m1, CNN_M1_PATH, device)

teacher = create_resnet18_teacher(num_classes=10)
teacher = load_weights(teacher, TEACHER_PATH, device)

print("Modelle erfolgreich geladen.")
print()

print_result(
    name="StudentKD20 Meilenstein 3",
    model=student,
    input_size=(3, 32, 32),
    accuracy=80.35,
    device=device
)

print_result(
    name="CNN V10 Meilenstein 1",
    model=cnn_m1,
    input_size=(3, 32, 32),
    accuracy=81.50,
    device=device
)

print_result(
    name="ResNet18 Teacher Meilenstein 2",
    model=teacher,
    input_size=(3, 224, 224),
    accuracy=90.99,
    device=device
)

print("-" * 70)
print("Speed-Test abgeschlossen.")
