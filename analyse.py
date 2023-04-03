# CS5330 Project 5: Recognition using Deep Network
# Task 2: Load a trained network and analyse the convoluion layer.
# Author: Yao Zhong, zhong.yao@northeastern.edu

# import statements
import sys
import torch
import numpy as np
import cv2 as cv
from torch import nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from model import MyNetwork


# main function
def main(argv):

    # Load the model and set the mode to evaluation
    model = torch.load('model.pth')
    model.eval()

    #print out the model in the terminal
    print(model)

    #get the weights of the first convolution layer(which is the first of the my_stack sequential)
    weights = model.my_stack[0].weight
    #visualize the weights
    figure=plt.figure(figsize=(9,8)) #8 inches 
    cols, rows = 4, 3
    with torch.no_grad():
        for i in range(weights.shape[0]):
            weight = weights[i,0]
            figure.add_subplot(rows, cols, i+1)
            plt.title(f"Filter: {i}")
            plt.axis("off")
            plt.imshow(weight.squeeze())
    plt.show()

    # Load the first image from the training data
    training_data = datasets.MNIST(
        root = "data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    # [0] is the first example, [0] is for the first entry(data), [0] is to make [1,28,28] to [28,28]
    img = training_data[0][0][0] 
    # convert to array so that can be processed by CV
    img_array = img.numpy()

    #loop for applying the filters and then plt the result
    filter_figure =plt.figure(figsize=(9,8)) #8 inches 
    cols, rows = 4, 5
    with torch.no_grad():
        for i in range(weights.shape[0]):
            # get the filter and then use filter2D to apply it on the image
            filter = weights[i,0]
            filter_array = filter.numpy()
            filter_img = cv.filter2D(img_array, -1, filter_array)
            # subplot the filtering itself
            filter_figure.add_subplot(rows, cols, 2*i+1)
            plt.title(f"Filter: {i}")
            plt.axis("off")
            plt.imshow(filter.squeeze(), cmap="gray")
            # subplot the filtered result
            filter_figure.add_subplot(rows, cols, 2*i+2)
            plt.title(f"Result: {i}")
            plt.axis("off")
            plt.imshow(filter_img.squeeze(), cmap="gray")
    plt.show()

    return

if __name__ == "__main__":
    main(sys.argv)