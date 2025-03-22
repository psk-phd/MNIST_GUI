import tkinter as tk
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import cv2
from models import *
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

# # Load trained model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
model.eval()

norm_mn = 0.1307    # Normalization mean
norm_std = 0.3081   # Normalization std
# GUI Class
GUI_Width = 1120; GUI_Height = 1120
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
        r = 40  # Brush size
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
        if model_name == 'SHLNN':
            img_tensor = img_tensor.view(1, -1)
        else:
            img_tensor = img_tensor.view(1, 1, 28, 28)
        with torch.no_grad():
            img_tensor = (img_tensor-norm_mn)/norm_std
            output = model(img_tensor)
            prediction = torch.argmax(output, dim=1).item()
        
        result_window = tk.Toplevel(self.root)
        result_window.title("Prediction")
        tk.Label(result_window, text=f"Predicted Digit: {prediction}", font=("Arial", 24)).pack()

    def clear_canvas(self):
        """Clears the canvas for new input."""
        self.canvas.delete("all")
        self.image = Image.new("L", (GUI_Width, GUI_Height), "black")
        self.draw = ImageDraw.Draw(self.image)

# Run GUI
root = tk.Tk()
app = DigitRecognizerApp(root)
root.mainloop()
