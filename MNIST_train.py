import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models import *
from torch.utils.data import random_split, DataLoader
# Define SHLNN Model
# class SHLNN(nn.Module):
#     def __init__(self, input_size=784, hidden_size=128, output_size=10):
#         super(SHLNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.softmax(self.fc2(x))
#         return x

# Load MNIST Dataset
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
# testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

device = 'cuda'
# Initialize model

# Loss and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# 🚀 Hyperparameters
# -------------------------------
learning_rate = 0.001
batch_size = 256
epochs = 200  # Max epochs
patience = 20  # Early stopping patience

# -------------------------------
# 🚀 Data Preprocessing (MNIST)
# -------------------------------
transform = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Standard MNIST mean/std
])

# Load full MNIST dataset
full_train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Split into 90% train, 10% validation
train_size = int(0.9 * len(full_train_set))
val_size = len(full_train_set) - train_size
train_set, val_set = random_split(full_train_set, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# -------------------------------
# 🚀 Model, Loss, and Optimizer
# -------------------------------
#hidden_layer_size = 128  # Changeable
model_name = 'CNN'
if model_name == 'SHLNN':
    model = SHLNN(hidden_size=hidden_layer_size).to(device)
elif model_name == 'LeNet':
    model = LeNet().to(device)
else:
    model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# -------------------------------
# 🚀 Training with Early Stopping
# -------------------------------
best_val_loss = float("inf")
epochs_no_improve = 0

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move to GPU/CPU

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation Phase
    model.eval()
    val_loss = 0
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    val_acc = correct / total * 100

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%")

    # Check for early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_name + "mnist.pth")  # Save best model
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break

print("Training complete. Best model saved as. " + model_name + "mnist.pth")

