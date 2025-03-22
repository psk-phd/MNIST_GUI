**Handwritten Digit Recognition using Spiking Neural Netowrks: An Interactive Demo**

An interactive handwritten digit (0-9) recognition GUI code utilizing spiking neural networks. Spiking neural networks utilize spike-trains instead of real numbers to transmit data, akin to how information is processed in the brain. In the demo, two windows open, one for character writing and another that shows an interactive display of how spikes flow in the network's last layer. Output probabilities are illustrated as 'dancing' histograms atop each output neuron corresponding to a digit between 0-9.  As the decision of the SNN gets more confident with time. one clear answer emerges. A video of the demo should make this clearer. 

https://github.com/user-attachments/assets/b6fa8c29-8beb-4620-b0e2-1693acb031d3

The demo plays upon running the file 'Final_Gui_char_recog_SNN.py'. Some other files can be played around with too. I will add their descriptions soon. A requirements.txt file has been provided. In addition, I recommend cloning spikingjelly repo (https://github.com/fangwei123456/spikingjelly) in your root directory. This was done because I was having difficulty installing spikingjelly and time was of the essence. If you have spikingjelly preinstalled, then you may need to tweak the headers in the .py files.

**Spiking Neurons and SNN brief overview**:
Spiking neurons are a biologically inspired type of artificial neuron that encode and process information through discrete spikes (action potentials) rather than continuous values. Unlike traditional artificial neural networks (ANNs), which rely on matrix multiplications and activations like ReLU or Sigmoid, spiking neural networks (SNNs) introduce the concept of time, making them more efficient and closer to how real neurons work.

Key Features of Spiking Neurons
    Event-driven computation: Neurons fire only when a threshold is reached, reducing unnecessary computations.
    Temporal dynamics: Information is encoded in the timing and frequency of spikes rather than just raw values.
    Energy efficiency: Ideal for low-power applications like edge computing and neuromorphic hardware.

Common Spiking Neuron Models
    Leaky Integrate-and-Fire (LIF) – Simplest model where the neuron accumulates input until a threshold is reached, then fires a spike.
    Integrate-and-Fire (IF) - LIF with no leak.
    Izhikevich Model – More biologically realistic, capable of capturing different neuron behaviors.
    Hodgkin-Huxley Model – A detailed, biophysical model based on ion channel dynamics.

Why Use Spiking Neurons?
    Closer to how the brain processes information
    Enables temporal processing (e.g., speech, event-based vision)
    Can be implemented on neuromorphic hardware (e.g., Loihi, SpiNNaker)

**SNN in this Project**:
This project uses a simple spiking convolutional neural network for the demo trained on MNIST dataset with augmentation (training files provided). The neuron chosen is IF. The SNN is converted from the trained ANN using convertors available in spikingjelly repo (refer spikingjelly docs for more info).

**Demo Details**:
Two windows pop up as shown in the video. The black canvas takes digit input, and upon clicking 'Predict', the simulation window comes alive with the dance of histograms. Spikes are animated along the synapses like 'beads on a string', and the color of synaptic edges indicates synaptic strengths. As synaptic currents combine in the output neurons, a softmax function converts the summed 'logits' into probabilities illustrated by the animated histogram bars. A scaling factor has been baked into the softmax exponent to better animate the histograms.

The input, in the form of handwritten digits, can either be provided using a mouse, or a commercially available graphic tablet like Wacom, which was used for the demo.

To clear the canvas, click 'Clear' and write a new digit. The animation then reruns, for a max of 100 timesteps. Please note you may need to restart the animation every hour and a half or so, because of the way the animation has been written. You can of course increase the max time further in 'Final_Gui_char_recog_SNN.py' by changing the numeral indicated in bold in the line animation.FuncAnimation(fig, update, frames=intperspike*(time_steps***150**), interval=100, blit=False, repeat=False).


