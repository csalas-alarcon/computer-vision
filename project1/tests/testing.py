import torch
import torchvision
import torchvision.transforms as transforms
import sys
import os

# Importar la clase SimpleNet desde tu archivo de entrenamiento
from src import SimpleNet

print("\n[TESTING]- Preparando datos y modelo...")
# 1. Recrear el transform
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 2. Cargar datos de prueba
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 3. Instanciar el modelo y cargar los pesos guardados
model = SimpleNet()
model.load_state_dict(torch.load('./models/our_model.pth', weights_only=True))
model.eval() # Modo evaluación

# 4. Evaluación
print("[TESTING]- Probando el modelo con datos nuevos...")
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Precisión en las 10,000 imágenes de prueba: {accuracy:.2f}%")