# CS5330 Project 5: Recognition using Deep Network
# Task 1: Build a network, train it and save it
# Author: Yao Zhong, zhong.yao@northeastern.edu

# import statements
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.models as models
import matplotlib.pyplot as plt


#load digit MNIST
def loadData():
    # Get the training data
    training_data = datasets.MNIST(
        root = "data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    #get the test data
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    return (training_data, test_data)

#Helper function that plots the first 6 images from the MNIST digits
def showExamples(training_data):
    figure=plt.figure(figsize=(8,6)) #8x6 inches window
    cols, rows = 3, 2
    for i in range(cols*rows):
        img, label = training_data[i]
        figure.add_subplot(rows, cols, i+1)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


# The structue of the model
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        # The stack of convolution layers and the followed relu and max pooling layers
        self.my_stack = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size = 5), #convlution layer 1, 28*28 to 10*24*24
            nn.MaxPool2d(2), #24*24 to 10*12*12
            nn.ReLU(), # relu layer
            nn.Conv2d(10, 20, kernel_size = 5), #10*12*12 to 20*8*8
            nn.Dropout(0.5), 
            nn.MaxPool2d(2), #20*8*8 to 20*4*4
            nn.ReLU()
        )
        # fully connected layers, for classification
        self.fc1=nn.Linear(320, 50) # 20*4*4 = 320
        self.fc2=nn.Linear(50, 10)

    # computes a forward pass for the network
    def forward(self, x):
        x = self.my_stack(x) # the convolution layers
        x = x.view(-1,320) # flatten according to the size and channels, 320 = 20x4x4
        x = F.relu(self.fc1(x)) # first fully connected layer with relu
        x = F.log_softmax(self.fc2(x), dim=1)  # log_softmax after the fc2
        return x

# The function that trians the network and plots the result of the trianing and testing
def train_network(train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs):
    #holder for the result of each epoch
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_dataloader.dataset) for i in range(epochs)]

    #call the train_loop and test_loop for each epoch
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, train_losses, train_counter, t)
        test_loop(test_dataloader, model, loss_fn, test_losses)
    print("Done!")

    # Plot the training and testing result
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

    return

# The train_loop for one epoch, the statistics are saved in losses and counter.
def train_loop(dataloader, model, loss_fn, optimizer, losses, counter, epocnIdx):
    #set the mode of the model being trainging, this will affect the dropout layer
    model.train()
    size = len(dataloader.dataset)
    for batchIdx, (data, target) in enumerate(dataloader):
        #forward
        pred = model(data)
        loss = loss_fn(pred, target)
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log in the terminal for each 10 batches
        if batchIdx % 10 == 0:
            loss, current = loss.item(), (batchIdx+1)*len(data)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            losses.append(loss)
            counter.append(batchIdx*len(data)+epocnIdx*size)

# The test_loop for one epoch, the statistics are saved in losses.
def test_loop(dataloader, model, loss_fn, losses):
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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# main function
def main(argv):
    # Task 1-b, make sure the model is repeatable
    random_seed = 47
    torch.manual_seed(random_seed)
    torch.backends.cudnn.enabled = False
    # Set the settings for the training
    learning_rate = 1e-2
    batch_size = 64
    epochs = 5
    
    #load traing data
    training_data, test_data = loadData()
    # Uncomment to show the first six images.
    # showExamples(training_data)

    # create dataloaders
    train_dataloader = DataLoader(training_data, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size, shuffle=True)
    
    # Get an instance of the network
    model = MyNetwork()
    
    # Loss fucntion
    loss_fn = nn.NLLLoss() # didnt choose CrossEntropyLoss because we already did the logsoftmax in the model
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    #start training
    train_network(train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs)
    #Save the trained network
    torch.save(model, 'model.pth')

    return

if __name__ == "__main__":
    main(sys.argv)