import torch
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

def display_mnist_image(index):
    # Load the MNIST dataset
    mnist = datasets.MNIST(root="./data", train=True, download=True)
    x_train, y_train = mnist.data, mnist.targets
    
    # Check if index is within range
    if index < 0 or index >= len(x_train):
        print("Index out of range. Please enter a valid index.")
        return
    
    # Display the image
    plt.imshow(x_train[index], cmap='gray')
    plt.title(f"Label: {y_train[index].item()}")
    plt.axis('off')
    plt.show()

# Example usage
display_mnist_image(654)
