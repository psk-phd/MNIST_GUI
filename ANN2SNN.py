import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import spikingjelly
from spikingjelly.activation_based import neuron, ann2snn
from spikingjelly import visualizing
import torchvision.transforms as transforms
from models import *
from torch.utils.data import random_split, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Initialize model
hidden_layer_size = 128  # Changeable

model_name = 'CNN'
if model_name == 'SHLNN':
    model = SHLNN(hidden_size=hidden_layer_size).to(device)
elif model_name == 'LeNet':
    model = LeNet().to(device)
else:
    model = CNN().to(device)
# -------------------------------
# 🚀 Hyperparameters
# -------------------------------
learning_rate = 0.0001
batch_size = 128
epochs = 100  # Max epochs
patience = 10  # Early stopping patience

model_fl = model_name + "mnist.pth"
model.load_state_dict(torch.load(model_fl, map_location=torch.device(device)))

transform = transforms.Compose([
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


model_converter = ann2snn.Converter(mode='max', dataloader=train_loader)
snn_model = model_converter(model)
torch.save(snn_model.state_dict(), model_fl + "_snn.pth")  # Save best model

print(snn_model)
