# CS5330 Project 5: Recognition using Deep Network
# Task 1: Load a trained network and run tests on it
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
# import the model structure
from model import MyNetwork

# task F, test on the examples on the dataset
def testOnDataset(model):
    # Load the test set
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    # start testing and store the predication in the list for latter plot
    predications = []
    with torch.no_grad():
        # test the first 10 images
        for i in range(10):
            data, target = test_data[i]
            print()
            pred = model(data)
            predications.append(pred.argmax(1).item())
            pred_array = pred.numpy()[0] # since pred.shape is [1,10], we need to only print out the second dimention, to get the array of 10 values
            # Print out the 10 values of each pred_array
            for value in pred_array:
                print(f"{value:<6.2f}", end=" ")
            print()
            #the real predication is the largest value index
            print(f"Predication: {pred.argmax(1).item()}, Target: {target}")
    
    # After testing the first 10 examples, plot out the first 9 of them
    figure=plt.figure(figsize=(9,12)) #8x12 inches 
    cols, rows = 3, 3
    for i in range(9):
        img, target = test_data[i]
        figure.add_subplot(rows, cols, i+1)
        plt.title(f"Predication: {predications[i]}")
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

#
def testOnNew(model):
    test_data = []
    # Pre-process the image so that it is similar with the digits from the datasets
    for i in range(10):
        # To grayscale
        img = cv.imread('newDigits/' + str(i) + '.jpg')
        gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
        # inverse and match intensity to [0~1]
        invert = cv.bitwise_not(gray)
        invert = invert/255 # to match the intensity with the test data, in which all values are 0~1
        # The tensor should be [1,28,28]
        addDim = invert[np.newaxis, :, :]
        tensor_trans = torch.tensor(addDim, dtype=torch.float32)
        test_data.append(tensor_trans)
    
    # start testing and store the predication in the list for latter plot
    predications = []
    with torch.no_grad():
        for i in range(10):
            data = test_data[i]
            print()
            pred = model(data)
            predications.append(pred.argmax(1).item())
            # print the pred values per requirement
            pred_array = pred.numpy()[0] # since pred.shape is [1,10], we need to only print out the second dimention, to get the array of 10 values
            for value in pred_array:
                print(f"{value:<6.2f}", end=" ")
            print()
            #the real predication is the largest value index
            print(f"Predication: {pred.argmax(1).item()}, Target: {i}")
            
    # After testing the first 10 examples, plot out them
    figure=plt.figure(figsize=(9,12)) #8 inches 
    cols, rows = 3, 4
    for i in range(10):
        img = test_data[i]
        figure.add_subplot(rows, cols, i+1)
        plt.title(f"Predication: {predications[i]}")
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()
    
# main function (yes, it needs a comment too)
def main(argv):
    
    # Load the saved model and set the mode to evaluation
    model = torch.load('model.pth')
    model.eval()

    # Task F, test on dataset
    testOnDataset(model)
    # Task G, test on new digits
    testOnNew(model)


    return

if __name__ == "__main__":
    main(sys.argv)