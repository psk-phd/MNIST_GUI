import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import spikingjelly
import sys
sys.path.append('/home/prabodh/Documents/MNIST_GUI/spikingjelly/spikingjelly/')
from spikingjelly.activation_based import neuron, ann2snn
from spikingjelly import visualizing
import torchvision.transforms as transforms
from models import *
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import numpy as np

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

def val(net, device, data_loader, T=None):
    net.eval().to(device)
    correct = 0.0
    total = 0.0
    if T is not None:
        corrects = np.zeros(T)
    with torch.no_grad():
        for batch, (img, label) in enumerate(tqdm(data_loader)):
            img = img.to(device)
            if T is None:
                out = net(img)
                correct += (out.argmax(dim=1) == label.to(device)).float().sum().item()
            else:
                for m in net.modules():
                    if hasattr(m, 'reset'):
                        m.reset()
                for t in range(T):
                    if t == 0:
                        out = net(img)
                    else:
                        out += net(img)
                    corrects[t] += (out.argmax(dim=1) == label.to(device)).float().sum().item()
            total += out.shape[0]
    return correct / total if T is None else corrects / total

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

model_fl = model_name + "mnist.pth"
model.load_state_dict(torch.load(model_fl, map_location=torch.device(device)))
model_converter = ann2snn.Converter(mode='max', dataloader=testloader)
snn_model = model_converter(model)
correct = 0
total = 0
T = 100
mode_max_accs = val(snn_model, device, testloader, T=T)
# with torch.no_grad():
#     for images, labels in testloader:
#         if model_name == 'SHLNN':
#             images = images.view(images.shape[0], -1).to(device)  # Flatten to (batch_size, 784)
#         else:
#             images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)  # Get predicted class
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

accuracy = 100 * mode_max_accs[-1]
print(f"Test Accuracy: {accuracy:.2f}%")
print(100 * mode_max_accs)