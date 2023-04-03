# CS5330 Project 5: Recognition using Deep Network
# Experiment 1: The effect of filter number and filter size
# Author: Yao Zhong, zhong.yao@northeastern.edu

# import statements
import sys
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.models as models
import matplotlib.pyplot as plt


#load MNIST fasion data
def loadData():
    # Load the training set
    training_data = datasets.FashionMNIST(
        root = "fashion_data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    # load the test set
    test_data = datasets.FashionMNIST(
        root="fashion_data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    return (training_data, test_data)

# class definitions, this class is in similar structure with model.py but parametized for automatic experiment
class ExNetwork(nn.Module):
    def __init__(self, filters_1, kernel_1, filters_2, kernel_2):
        super(ExNetwork, self).__init__()
        #calculate the final linear size for later use, did it step by step just to make it clear
        self.after_pool1 = (28 - kernel_1 + 1)//2
        self.after_pool2 = (self.after_pool1 - kernel_2 + 1)//2
        self.linear_size = filters_2*self.after_pool2*self.after_pool2
        # mainstructure of the model
        self.my_stack = nn.Sequential(
            nn.Conv2d(1, filters_1, kernel_size = kernel_1), #28*28 to filters_1*(28-kernel_1+1)*(28-kernel_1+1)
            nn.MaxPool2d(2), #filters_1*(28-kernel_1+1)*(28-kernel_1+1) to filters_1*((28-kernel_1+1)/2)*((28-kernel_1+1)/2)
            nn.ReLU(),
            nn.Conv2d(filters_1, filters_2, kernel_size = kernel_2), 
            # filters_1*((28-kernel_1+1)/2)*((28-kernel_1+1)/2) to filters_2*((28-kernel_1+1)/2 - kernel_2 + 1)*((28-kernel_1+1)/2 - kernel_2 + 1)
            nn.Dropout(0.5),
            nn.MaxPool2d(2), #reduce to linear_size
            nn.ReLU()
        )
        # fully connected layers
        self.fc1=nn.Linear(self.linear_size, 50) # filters_2*(((28-kernel_1+1)/2 - kernel_2 + 1)/2)*(((28-kernel_1+1)/2 - kernel_2 + 1)/2)
        self.fc2=nn.Linear(50, 10)

    # computes a forward pass for the network
    def forward(self, x):
        x = self.my_stack(x) # the convolution layers
        x = x.view(-1,self.linear_size) # flatten to linear
        x = F.relu(self.fc1(x)) # first fully connected layer with relu
        x = F.log_softmax(self.fc2(x), dim=1) # log_softmax after all layers
        return x

# The function that trians and test the network, but this time will not log the training process, only the test lost and accuracy is logged
def train_network(train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs):
    #holder for the result of each epoch test
    test_loss = []
    test_corrects = []
    #call the train_loop and test_loop for each epoch
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn, test_loss, test_corrects)
    print("Done!")
    # return the test result of the last epoch of the training as the result of the variant model
    return (test_loss[-1],test_corrects[-1])

# The train_loop for one epoch,
def train_loop(dataloader, model, loss_fn, optimizer):
    #set the mode of the model being trainging, this will affect the dropout layer
    model.train()
    for batchIdx, (data, target) in enumerate(dataloader):
        #forward
        pred = model(data)
        loss = loss_fn(pred, target)
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# The test_loop for one epoch, the statistics are saved in losses and corrects
def test_loop(dataloader, model, loss_fn, losses, corrects):
    # set the mode of the model being testing, this will affect the dropout layer
    model.eval()
    # variables for accuracy computing
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    #get the test loss for each batch
    with torch.no_grad():
        for data, target in dataloader:
            pred = model(data)
            test_loss += loss_fn(pred, target).item()
            correct += (pred.argmax(1) == target).type(torch.float).sum().item()

    # calculate and log the accuracy of test
    test_loss /= num_batches
    losses.append(test_loss)
    correct /= size
    corrects.append(correct)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# helper function that draws the graphs for the experiment
def showGraph(results, filter_numbers, combos):
    # there will be 9 combinations and bar width is 0.15
    ind = np.arange(9)
    width = 0.15
    colors = ['red','green','blue','yellow']

    # generate the legends for bars
    legends = []
    for value in filter_numbers:
        legends.append(f"L1: {value}, L2: {value*2}")

    # plot diffrent bars
    for i in range(4):
        plt.bar(ind + i*width, results[i], width, color= colors[i])

    # add legends and labels
    plt.legend(legends, loc='upper right')
    plt.xticks(ind+width, combos)
    plt.ylabel('negative log likelihood loss')

    plt.show()


# main function
def main(argv):
    # make sure the model is repeatable
    random_seed = 47
    torch.manual_seed(random_seed)
    torch.backends.cudnn.enabled = False

    # Set the settings for the training
    learning_rate = 1e-2
    batch_size = 64
    epochs = 5
    loss_fn = nn.NLLLoss() # didnt choose CrossEntropyLoss because we already did the logsoftmax in the model

    # Load data for the experiments
    training_data, test_data = loadData()
    train_dataloader = DataLoader(training_data, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size, shuffle=True)

    # filter size and numbers options
    filter_sizes = [3,5,7]
    filter_numbers = [5, 10, 15, 20]

    # containers for the loss and accuracy result, for latter plot and print
    size_loss = [] #loss by the filter number dimention, will be 4x9
    size_correct = []  #similar with size_loss but store the accuracy
    best_loss = None #store the best loss
    best_combo = [] #store the best combination 
    combos = [] #different combinations of kernel size(total 9 of them)
    # gnerating those combinations, for latter plot
    for kernel_1 in filter_sizes:
            for kernel_2 in filter_sizes:
                combos.append(f"K{kernel_1},K{kernel_2}")

    # main loop for the experiment, trians the model with different parameters
    for filters_1 in filter_numbers:
        # store the loss and accuracy of this filter number level, will be array of length 9
        losses = []
        corrects = []
        # iterate through kernel size combinations
        for kernel_1 in filter_sizes:
            for kernel_2 in filter_sizes:
                # instantiate the network
                model = ExNetwork(filters_1, kernel_1, 2*filters_1, kernel_2)
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
                # train and test the network and record the result
                test_loss, test_correct = train_network(train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs)
                losses.append(test_loss)
                corrects.append(test_correct)
                #  find the model parameter with minimum loss
                if best_loss == None or test_loss<best_loss:
                    best_loss = test_loss
                    best_combo = [filters_1, kernel_1, 2*filters_1, kernel_2]
        # store each filter numbers result
        size_loss.append(losses)
        size_correct.append(corrects)

    # draw the average loss graph and accuracy graph
    showGraph(size_loss, filter_numbers, combos)
    showGraph(size_correct, filter_numbers, combos)

    # print out the statistics
    print(size_loss)
    print(size_correct)
    print(best_combo)
    return

if __name__ == "__main__":
    main(sys.argv)