# ./src/training.py 

# Torch
import torch 
import torch.nn as nn # Neural Networks
import torch.optim as optim # Algos for Optimization
# Torchvision
import torchvision
import torchvision.transforms as transforms # Tweaks Images before use


# Minimal Neural Network - Inherits from Pytorch's base Class
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # CIFAR-10 Images (3x32x32) (RGB x X Pixels x Y Pixels)
        self.fc1 = nn.Linear(3 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def main():
    print("[TRAINING]- Downloading Data")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    print("[TRAINING]- Instantiating Model, Criterion, Optimizer")
    model = SimpleNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("[TRAINING]- Beginning 1-Epoch Training ")
    for i, (images, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"batch {i}, Loss: {loss.item():.4f}")

    print("[TRAINING]- Finished Training")

    torch.save(model.state_dict(), './models/our_model.pth')
    print("Model saved to ./models/our_model.pth")

if __name__ == '__main__':
    main()