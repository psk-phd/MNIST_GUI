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
from PIL import Image, ImageDraw
import cv2
import tkinter as tk

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
single_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

def valone(net, device, img, T=None):
    #img, target = next(iter(single_loader))
    net.eval().to(device)
    #print('data label is', target)
    with torch.no_grad():
        outtensor = torch.zeros(T,10).to(device)
        #img = img.to(device)
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
intperspike = 4
norm_mn = 0.1307    # Normalization mean
norm_std = 0.3081   # Normalization std

#%%
hook_output = torch.zeros(T,32)
indx = 0

def hook_fn(module, input, output):
    #print('Hooked!!')
    #print(output.shape)
    global hook_output
    global indx
    indices = torch.randperm(output.shape[1])
    hook_vals = output.flatten()[indices]
    hook_op = hook_vals>0
    hook_output[indx%T]= hook_op*1
    indx+=1
    #print(hook_output)
    return output
# snn_tailor_module = getattr(snn_model, 'snn tailor')
# if_node = getattr(getattr(snn_tailor_module, '2'), '1')
# hook = if_node.register_forward_hook(hook_fn)
minalfa = 0.1
with torch.no_grad():   
    if_node = getattr(snn_model.network, '12')
    hook = if_node.register_forward_hook(hook_fn)
    Lin_Layer = getattr(snn_model.network, '13').weight
    Lin_Layer = Lin_Layer + minalfa
    Lin_scaled = Lin_Layer/max(Lin_Layer.flatten())
    Lin_scaled[Lin_scaled<minalfa] = minalfa
    Lin_scaled = Lin_scaled.detach().cpu().numpy()


#mode_max_accs = val(snn_model, device, testloader, T=T)
#%%
GUI_Width = 980; GUI_Height = 980

def hist_anim(outtensor_prob, hook_output, start_anim=True):
    num_hidden = 32   # Hidden neurons
    num_output = 10  # Output neurons (MNIST classes)
    time_steps = T # Duration of animation

    # Generate random spike activity for hidden layer (Binary: 0 or 1)
    spike_prob = 0.05  # 🔹 Adjust this to control spike density
    #hidden_spikes = (np.random.rand(time_steps, num_hidden) < spike_prob).astype(int)
    hidden_spikes = hook_output
    print(hidden_spikes.shape)


    # Random synaptic weights (for visualization only, not real training)
    synaptic_weights = np.random.rand(num_hidden, num_output)

    # Compute output neuron activations over time
    output_activations = np.zeros((time_steps, num_output))
    # for t in range(time_steps):
    #     output_activations[t] = np.dot(hidden_spikes[t], synaptic_weights)  # Weighted sum
    outtensor_prob_scores = torch.softmax(0.01*outtensor_prob,1)
    output_activations = outtensor_prob_scores.cpu().numpy()

    # Normalize output activations for histogram scaling
    output_histogram = np.zeros((time_steps, num_output))
    for t in range(time_steps):
        max_val = np.max(output_activations[t]) if np.max(output_activations[t]) != 0 else 1
        output_histogram[t] = output_activations[t] / max_val  # Normalize for scaling

    # Live prediction (most active neuron)
    predictions = np.argmax(outtensor_prob, axis=1)

    # -------------------------------
    # 🚀 Matplotlib Figure Setup
    # -------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    # Layout: Synapses at bottom, histogram at top
    ax.set_xlim(-1, num_output)  
    ax.set_ylim(-0.7, 1.5)  # Extra space for histograms above the network
    ax.set_axis_off()
    # Hidden & Output Neuron Positions (Fixed Layout)
    hidstart = -0.5; hidend = num_output - 0.5
    hidden_positions = np.linspace(hidstart, hidend, num_hidden)  # Evenly spread hidden nodes
    output_positions = np.arange(num_output)  # Output nodes at the top

    honodes = [-0.5,0]
    # Draw static network structure (Synapses & Neurons)
    synapse_lines = []
    for h in range(num_hidden):
        for o in range(num_output):
            line = plt.plot([hidden_positions[h], output_positions[o]], honodes, "gray", alpha=minalfa)  # Fixed synapses
            synapse_lines.append(line)

    hidden_nodes = ax.scatter(hidden_positions, [honodes[0]] * num_hidden, s=100, color="blue", label="Hidden Neurons")
    output_nodes = ax.scatter(output_positions, [honodes[1]] * num_output, s=150, color="red", label="Output Neurons")
    
    for ctr in output_positions:
        i = ctr
        j = honodes[1]
        corr = 0
        ax.annotate(f'{int(ctr)}', xy=(i+corr,j+corr), fontsize=10, ha='center', va='center')
        #ctr+=1

    # Bars for output neuron histogram (Above the network)
    bars = ax.bar(output_positions, np.zeros(num_output), color="royalblue", width=0.6)

    # Moving spike markers (Beads traveling along synapses)
    num_spikes = num_hidden * num_output  # One potential spike per synapse
    spike_markers = [ax.plot([], [], "|", color="black", alpha=0)[0] for _ in range(num_spikes)]

    # Text label for prediction
    prediction_text = ax.text(num_output // 2, 1.2, "Prediction: ?", fontsize=14, ha="center")
    time_text = ax.text(num_output // 2, 1.4, "Timestep: ?", fontsize=14, ha="center")
 
    # -------------------------------
    # 🚀 Animation Function
    # -------------------------------
    def update(frame):
        """ Updates spike movements, histogram bars, and prediction per frame. """
        curr_tim = int(np.floor(frame/intperspike))
        # Update histogram bars
        for i, bar in enumerate(bars):
            bar.set_height(output_histogram[curr_tim][i])

        for line in synapse_lines:
            if isinstance(line[0], plt.Line2D):  # Ensure it's a Line2D object
                line[0].set_alpha(minalfa)
                #line[0].set_color('gray')
                line[0].set_linewidth(1)
        
        # Animate spikes traveling over synapses like beads
        for i, spike_marker in enumerate(spike_markers):
            h = i // num_output  # Hidden neuron index
            o = i % num_output   # Output neuron index
            synapse_index = h * num_output + o

            if hidden_spikes[curr_tim, h]:  # If a spike occurs in this hidden neuron
            #if hidden_spikes[h]:
                alpha_value = 1 - (frame % intperspike) / intperspike  # Gradual fading effect
                t = (frame % intperspike) / intperspike  # Progress along synapse
                
                x = (1 - t) * hidden_positions[h] + t * output_positions[o]  # Interpolate position
                y = (1 - t) * (honodes[0]) + t * honodes[1]  # Interpolate height
                #plt.plot([hidden_positions[h], output_positions[o]], honodes, "gray", alpha=1)

                spike_marker.set_data([x], [y])
                spike_marker.set_alpha(alpha_value)
                alfa = Lin_scaled[o,h]

                if 0 <= synapse_index < len(synapse_lines):  # Ensure valid index
                    if isinstance(synapse_lines[synapse_index][0], plt.Line2D):  # Ensure it's a line object
                        synapse_lines[synapse_index][0].set_alpha(alfa)
                        synapse_lines[synapse_index][0].set_color('salmon')
                        synapse_lines[synapse_index][0].set_linewidth(2)
            else:
                spike_marker.set_alpha(max(0, spike_marker.get_alpha() - 0.1))  # Fade out gradually
                # synapse_lines[synapse_index].set_alpha(0.3)

        # Update prediction text
        prediction_text.set_text(f"Prediction: {predictions[curr_tim]}")
        time_text.set_text(f"Time: {curr_tim}")

        return bars, spike_markers, prediction_text

    # Run animation
    if start_anim:
        ani = animation.FuncAnimation(fig, update, frames=intperspike*time_steps, interval=200, blit=False, repeat=False)
        plt.show()


class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognition")

        # Canvas for drawing
        self.canvas = tk.Canvas(root, width=GUI_Width, height=GUI_Height, bg="black")
        self.canvas.pack()

        # Buttons
        self.button_predict = tk.Button(root, text="Predict", command=self.predict)
        self.button_predict.pack(side=tk.LEFT, padx=10, pady=10)

        self.button_clear = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.button_clear.pack(side=tk.RIGHT, padx=10, pady=10)

        # Initialize drawing
        self.image = Image.new("L", (GUI_Width, GUI_Height), "black")
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.draw_digit)

    def draw_digit(self, event):
        """Draws on the canvas when the user moves the mouse while holding the left button."""
        x, y = event.x, event.y
        r = 50  # Brush size
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill="white", outline="white")

    def preprocess_image(self, margin=4):
        """Prepares the drawn image for model prediction with adjustable margins."""
        img = np.array(self.image)

        # Convert to white digit on black background
        #img = 255 - img  

        # Find bounding box and crop
        coords = cv2.findNonZero(img)
        x, y, w, h = cv2.boundingRect(coords)
        img_cropped = img[y:y+h, x:x+w]

    #     coords = np.where(img > 0)
    #     if coords[0].size == 0:  # If no drawing detected, return a blank image
    #         return torch.zeros(1, 784)

    #     y_min, y_max = np.min(coords[0]), np.max(coords[0])
    #     x_min, x_max = np.min(coords[1]), np.max(coords[1])

    # # Crop the image tightly
    #     img_cropped = img[y_min:y_max+1, x_min:x_max+1]

        # Resize digit to (28 - 2*margin) x (28 - 2*margin)
        new_size = 28 - 2 * margin
        img_resized = cv2.resize(img_cropped, (new_size, new_size), interpolation=cv2.INTER_AREA)

        # Create a 28x28 black image and paste the resized digit
        img_padded = np.zeros((28, 28), dtype=np.uint8)
        img_padded[margin:margin+new_size, margin:margin+new_size] = img_resized

        # cv2.imshow("Preprocessed Image", img_padded)
        # cv2.waitKey(0)  # Wait for a key press to close
        # cv2.destroyAllWindows()

        # Normalize
        img_padded = img_padded.astype(np.float32) / 255.0
        #img_padded = img_padded.reshape(1, 784)  # Flatten

        return torch.tensor(img_padded)


    def predict(self):
        """Runs the model on the processed image and displays the prediction."""
        img_tensor = self.preprocess_image().to(device)
        #hook_output = torch.zeros(T,32)

        
        if model_name == 'SHLNN':
            img_tensor = img_tensor.view(1, -1)
        else:
            img_tensor = img_tensor.view(1, 1, 28, 28)
        with torch.no_grad():
            img_tensor = (img_tensor-norm_mn)/norm_std
            #output = model(img_tensor)
            #prediction = torch.argmax(output, dim=1).item()
        
            singleop, outtensor = valone(snn_model, device, img_tensor, T=T)


            #outtensor_prob = torch.softmax(0.01*outtensor,1)
            outtensor_prob = outtensor
            #print(outtensor_prob.shape)
            #outtensor_prob = outtensor

        hist_anim(outtensor_prob, hook_output)
        print(hook_output)
        # result_window = tk.Toplevel(self.root)
        # result_window.title("Prediction")
        # tk.Label(result_window, text=f"Predicted Digit: {prediction}", font=("Arial", 24)).pack()

    def clear_canvas(self):
        """Clears the canvas for new input."""
        self.canvas.delete("all")
        self.image = Image.new("L", (GUI_Width, GUI_Height), "black")
        self.draw = ImageDraw.Draw(self.image)

# Run GUI
root = tk.Tk()
#hist_anim(torch.zeros(T,10), torch.zeros(T,32), start_anim=False)
app = DigitRecognizerApp(root)
root.mainloop()


#%%

# %%
