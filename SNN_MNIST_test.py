import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import spikingjelly
import sys
from models import *
sys.path.append('/home/prabodh/Documents/MNIST_GUI/spikingjelly/spikingjelly/')
from spikingjelly.activation_based import neuron, ann2snn
from spikingjelly import visualizing
import torchvision.transforms as transforms

from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
single_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
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


def valone(net, device, data_loader, T=None):
    img, target = next(iter(single_loader))
    net.eval().to(device)
    print('data label is', target)
    with torch.no_grad():
        outtensor = torch.zeros(T,10).to(device)
        img = img.to(device)
        if T is None:
            out = net(img)
        else:
            for m in net.modules():
                if hasattr(m, 'reset'):
                    m.reset()
            for t in range(T):
                if t == 0:
                    out = net(img)
                else:
                    out += net(img)
                    #print(out)
                    outtensor[t] = out
    return out, outtensor
    
#%%
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
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


#mode_max_accs = val(snn_model, device, testloader, T=T)
#%%
hook_output = None

def hook_fn(module, input, output):
    #print('Hooked!!')
    #print(output.shape)
    indices = torch.randperm(output.shape[1])[:8]
    hook_vals = output.flatten()[indices]
    hook_output = hook_vals>0
    hook_output= hook_output*1
    print(hook_output)
    return output
# snn_tailor_module = getattr(snn_model, 'snn tailor')
# if_node = getattr(getattr(snn_tailor_module, '2'), '1')
# hook = if_node.register_forward_hook(hook_fn)
if_node = getattr(snn_model.network, '12')
hook = if_node.register_forward_hook(hook_fn)

singleop, outtensor = valone(snn_model, device, single_loader, T=T)
outtensor_prob = torch.softmax(outtensor,1)

#%%
num_hidden = 32   # Hidden neurons
num_output = 10  # Output neurons (MNIST classes)
time_steps = 100 # Duration of animation

# Generate random spike activity for hidden layer (Binary: 0 or 1)
spike_prob = 0.05  # 🔹 Adjust this to control spike density
hidden_spikes = (np.random.rand(time_steps, num_hidden) < spike_prob).astype(int)


# Random synaptic weights (for visualization only, not real training)
synaptic_weights = np.random.rand(num_hidden, num_output)

# Compute output neuron activations over time
output_activations = np.zeros((time_steps, num_output))
# for t in range(time_steps):
#     output_activations[t] = np.dot(hidden_spikes[t], synaptic_weights)  # Weighted sum

output_activations = outtensor_prob.cpu().numpy()

# Normalize output activations for histogram scaling
output_histogram = np.zeros((time_steps, num_output))
for t in range(time_steps):
    max_val = np.max(output_activations[t]) if np.max(output_activations[t]) != 0 else 1
    output_histogram[t] = output_activations[t] / max_val  # Normalize for scaling

# Live prediction (most active neuron)
predictions = np.argmax(output_histogram, axis=1)

# -------------------------------
# 🚀 Matplotlib Figure Setup
# -------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

# Layout: Synapses at bottom, histogram at top
ax.set_xlim(-1, num_output)  
ax.set_ylim(-0.7, 1.5)  # Extra space for histograms above the network
ax.set_axis_off()
# Hidden & Output Neuron Positions (Fixed Layout)
hidden_positions = np.linspace(0, num_output - 1, num_hidden)  # Evenly spread hidden nodes
output_positions = np.arange(num_output)  # Output nodes at the top

honodes = [-0.5,0]
# Draw static network structure (Synapses & Neurons)
synapse_lines = []
for h in range(num_hidden):
    for o in range(num_output):
        line = plt.plot([hidden_positions[h], output_positions[o]], honodes, "gray", alpha=0.3)  # Fixed synapses
        synapse_lines.append(line)

hidden_nodes = ax.scatter(hidden_positions, [honodes[0]] * num_hidden, s=100, color="blue", label="Hidden Neurons")
output_nodes = ax.scatter(output_positions, [honodes[1]] * num_output, s=150, color="red", label="Output Neurons")

# Bars for output neuron histogram (Above the network)
bars = ax.bar(output_positions, np.zeros(num_output), color="royalblue", width=0.6)

# Moving spike markers (Beads traveling along synapses)
num_spikes = num_hidden * num_output  # One potential spike per synapse
spike_markers = [ax.plot([], [], "|", color="black", linewidth = 2, alpha=0)[0] for _ in range(num_spikes)]

# Text label for prediction
prediction_text = ax.text(num_output // 2, 1.2, "Prediction: ?", fontsize=14, ha="center")
time_text = ax.text(num_output // 2, 1.4, "Timestep: ?", fontsize=14, ha="center")

# -------------------------------
# 🚀 Animation Function
# -------------------------------
def update(frame):
    """ Updates spike movements, histogram bars, and prediction per frame. """
    
    # Update histogram bars
    for i, bar in enumerate(bars):
        bar.set_height(output_histogram[frame][i])
    
    # Animate spikes traveling over synapses like beads
    for i, spike_marker in enumerate(spike_markers):
        h = i // num_output  # Hidden neuron index
        o = i % num_output   # Output neuron index

        if hidden_spikes[frame, h]:  # If a spike occurs in this hidden neuron
            alpha_value = 1 - (frame % 10) / 10  # Gradual fading effect
            t = (frame % 10) / 10  # Progress along synapse
            
            x = (1 - t) * hidden_positions[h] + t * output_positions[o]  # Interpolate position
            y = (1 - t) * (honodes[0]) + t * honodes[1]  # Interpolate height

            spike_marker.set_data([x], [y])
            spike_marker.set_alpha(alpha_value)
        else:
            spike_marker.set_alpha(max(0, spike_marker.get_alpha() - 0.1))  # Fade out gradually

    # Update prediction text
    prediction_text.set_text(f"Prediction: {predictions[frame]}")
    time_text.set_text(f"Time: {frame}")

    return bars, spike_markers, prediction_text

# Run animation
ani = animation.FuncAnimation(fig, update, frames=time_steps, interval=200, blit=False, repeat=False)
plt.show()
# %%
