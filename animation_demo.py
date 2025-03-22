import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -------------------------------
# 🚀 Visualization Parameters
# -------------------------------
num_hidden = 8   # Hidden neurons
num_output = 10  # Output neurons (MNIST classes)
time_steps = 100 # Duration of animation

# Generate random spike activity for hidden layer (Binary: 0 or 1)
spike_prob = 0.05  # 🔹 Adjust this to control spike density
hidden_spikes = (np.random.rand(time_steps, num_hidden) < spike_prob).astype(int)


# Random synaptic weights (for visualization only, not real training)
synaptic_weights = np.random.rand(num_hidden, num_output)

# Compute output neuron activations over time
output_activations = np.zeros((time_steps, num_output))
for t in range(time_steps):
    output_activations[t] = np.dot(hidden_spikes[t], synaptic_weights)  # Weighted sum

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

# Hidden & Output Neuron Positions (Fixed Layout)
hidden_positions = np.linspace(0, num_output - 1, num_hidden)  # Evenly spread hidden nodes
output_positions = np.arange(num_output)  # Output nodes at the top

honodes = [-0.5,0]
# Draw static network structure (Synapses & Neurons)
for h in range(num_hidden):
    for o in range(num_output):
        plt.plot([hidden_positions[h], output_positions[o]], honodes, "gray", alpha=0.3)  # Fixed synapses

hidden_nodes = ax.scatter(hidden_positions, [honodes[0]] * num_hidden, s=100, color="blue", label="Hidden Neurons")
output_nodes = ax.scatter(output_positions, [honodes[1]] * num_output, s=150, color="red", label="Output Neurons")

# Bars for output neuron histogram (Above the network)
bars = ax.bar(output_positions, np.zeros(num_output), color="royalblue", width=0.6)

# Moving spike markers (Beads traveling along synapses)
num_spikes = num_hidden * num_output  # One potential spike per synapse
spike_markers = [ax.plot([], [], "o", color="black", alpha=0)[0] for _ in range(num_spikes)]

# Text label for prediction
prediction_text = ax.text(num_output // 2, 1.2, "Prediction: ?", fontsize=14, ha="center")

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

            spike_marker.set_data(x, y)
            spike_marker.set_alpha(alpha_value)
        else:
            spike_marker.set_alpha(max(0, spike_marker.get_alpha() - 0.1))  # Fade out gradually

    # Update prediction text
    prediction_text.set_text(f"Prediction: {predictions[frame]}")

    return bars, spike_markers, prediction_text

# Run animation
ani = animation.FuncAnimation(fig, update, frames=time_steps, interval=100, blit=False)
plt.show()
