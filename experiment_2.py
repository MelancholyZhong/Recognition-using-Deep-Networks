# CS5330 Project 5: Recognition using Deep Network
# Experiment 2: The effect of dropout rate and dropout layer number
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

# inherit the functions from experiment_! 
from experiment_1 import loadData, train_network, train_loop, test_loop

# class definitions, this class is in similar structure with model.py but parametized for automatic experiment
class ExNetwork(nn.Module):
    def __init__(self, dropout_rate, add_layer=False):
        super(ExNetwork, self).__init__()
        # mainstructure of the model
        self.my_stack = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size = 5), # convolution layer 1, use the optimal information from experiemnt 1
            nn.MaxPool2d(2), 
            nn.ReLU(),
            nn.Conv2d(20, 40, kernel_size = 7), 
            # the dropout rate is parametized for the experiment
            nn.Dropout(dropout_rate),
            nn.MaxPool2d(2), 
            nn.ReLU()
        )
        # if to add another dropout layer
        self.add_layer = add_layer
        # fully connected layers
        self.fc1=nn.Linear(360, 50)
        self.fc2=nn.Linear(50, 10)

    # computes a forward pass for the network
    def forward(self, x):
        x = self.my_stack(x) # the convolution layers
        x = x.view(-1,360) # flatten to linear
        # decide wether to add another dropout layer after the first fully connected layer
        if self.add_layer:
            x = F.relu(F.dropout(self.fc1(x))) #add a dropout layer
        else:
            x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)# log_softmax after all layers
        return x     

# helper function that draws the graphs for the experiment
def showGraph(results, dropout_rates, ylabel):
    # there will be 10 combinations by the dropout rate levels
    ind = np.arange(10)
    width = 0.2
    colors = ['red','green']

    # generate the legends for bars
    legends = ["No Add Layer", "Add Layer"]

    # plot diffrent bars
    for i in range(2):
        plt.bar(ind + i*width, results[i], width, color= colors[i])

    # add legends and labels
    plt.legend(legends, loc='upper right')
    plt.xticks(ind+width, dropout_rates)
    plt.ylabel(ylabel)

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

    # options on each dimension
    dropout_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    add_layer = [False, True]

    # containers for the loss and accuracy result, for latter plot and print
    combo_loss = [] #loss by the add_layer dimention, will be 2x10
    combo_correct = [] #similar with loss but store the accuracy
    best_loss = None #store the best loss
    best_combo = [] #store the best combination 
    
    # main loop for the experiment, trians the model with different parameters
    for _if_add in add_layer:
        # store the loss and accuracy of this add_layer condition, will be array of length 10
        losses = []
        corrects = []
        # iterate through different dropout rates
        for rate in dropout_rates:
            # instantiate the network
            model = ExNetwork(rate, _if_add)
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            # train and test the network and record the result
            test_loss, test_correct = train_network(train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs)
            losses.append(test_loss)
            corrects.append(test_correct)
            #  find the model parameter with minimum loss
            if best_loss == None or test_loss<best_loss:
                best_loss = test_loss
                best_combo = [_if_add, rate]
        # store each add_layer option's result
        combo_loss.append(losses)
        combo_correct.append(corrects)

    # draw the average loss graph and accuracy graph
    showGraph(combo_loss,dropout_rates, "negative log likelihood loss")
    showGraph(combo_correct,dropout_rates, "accuracy")
    
    # print out the statistics
    print(combo_loss)
    print(combo_correct)
    print(best_combo)
    return

if __name__ == "__main__":
    main(sys.argv)