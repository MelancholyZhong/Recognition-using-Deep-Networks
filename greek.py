# CS5330 Project 5: Recognition using Deep Network
# Task 3 and Extension: Transfer learning to the greek letters
# Author: Yao Zhong, zhong.yao@northeastern.edu

# import statements
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.models as models
import matplotlib.pyplot as plt

from model import MyNetwork

# greek data set transform
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )

#load and transform the greek letters from the specified dataset
def loadData(training_set_path):
    # DataLoader for the Greek data set
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder( training_set_path,
                                          transform = torchvision.transforms.Compose( [torchvision.transforms.ToTensor(),
                                                                                       GreekTransform(),
                                                                                       torchvision.transforms.Normalize(
                                                                                           (0.1307,), (0.3081,) ) ] ) ),
        batch_size = 5,
        shuffle = True )
    return greek_train

# The function that trians the network and plots the result of the trianing
def train_network(train_dataloader, model, loss_fn, optimizer, epochs):
    train_losses = []
    train_counter = []
    
    #call the train_loop for each epoch and store the loss in an array
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, train_losses, train_counter, t)
    print("Done!")

    # Plot the training error
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.legend(['Train Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

    return

# The train_loop for one epoch, the statistics are saved in losses and counter.
def train_loop(dataloader, model, loss_fn, optimizer, losses, counter, epocnIdx):
    #set the mode of the model being trainging, this will affect the dropout layer
    model.train()
    size = len(dataloader.dataset)
    correct = 0
    for batchIdx, (data, target) in enumerate(dataloader):
        #forward
        pred = model(data)
        loss = loss_fn(pred, target)
        correct += (pred.argmax(1) == target).type(torch.float).sum().item()
         #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # log in the terminal for each 2 batches(10 images)
        if batchIdx % 2 == 0:
            loss, current = loss.item(), (batchIdx+1)*len(data)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            #save the loss data and its index for final plot
            losses.append(loss)
            counter.append(batchIdx*len(data)+epocnIdx*size)
    # calculate the accuracy of each training epoch
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%\n")
        
# test with the new greek letters that I wrote
def test_on_new(dataloader, model):
    # set the mode to evaluation
    model.eval()
    # Use this array, so that have the 0-alpha, 1-beta, 2-gamma and so on
    # create the plot window
    size = len(dataloader.dataset)
    figure=plt.figure(figsize=(9,9)) #9x9 inches 
    cols, rows = 3, (size+2)//3
    # main loop which test a image and plot it with the predication result
    with torch.no_grad():
        for sub_idx, (img,target)  in enumerate(dataloader.dataset):
            pred = model(img)
            # plot the sub plot
            figure.add_subplot(rows, cols, sub_idx+1)
            plt.title(f"Pred: {pred.argmax(1).item()}, Target:{target}")
            plt.axis("off")
            plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


# main function
def main(argv):
    # handel argv
    commands = ['original', 'additional_v1', 'additional_v2', 'extended']
    if len(argv) < 2:
        mode = 'original' #by default we run the task 3
    else:
        mode = argv[1]
    if mode not in  commands:
        print("Command not accepted")
        exit(-1)

    # make sure the model is repeatable
    random_seed = 47
    torch.manual_seed(random_seed)
    torch.backends.cudnn.enabled = False

    # Set the settings for the training
    learning_rate = 1e-2
    epochs = 16
    
    # load the trianed model
    model = torch.load('model.pth')
    # freezes the parameters for the whole network
    for param in model.parameters():
        param.requires_grad = False
    #replace the last layer with a new 3-nodes layer
    if mode != 'extended': 
        model.fc2 = nn.Linear(50,3)
    else:
        model.fc2 = nn.Linear(50, 5)

    # decide which train and test set to use according to the mode
    if mode == 'original':
        greek_train_path = './greek_letters/greek_train'
        greek_test_path  = './greek_letters/greek_test'
    elif mode == 'additional_v1':
        greek_train_path = './greek_letters/additional_greek_train_v1'
        greek_test_path  = './greek_letters/greek_test'
    elif mode == 'additional_v2':
        greek_train_path = './greek_letters/additional_greek_train_v2'
        greek_test_path  = './greek_letters/greek_test'
    else:
        greek_train_path = './greek_letters/extended_greek_train'
        greek_test_path  = './greek_letters/extended_greek_test'

    # Load the training data and test data
    greek_train_loader = loadData(greek_train_path)
    greek_test_loader = loadData(greek_test_path)

     # Loss fucntion
    loss_fn = nn.NLLLoss() # didnt choose CrossEntropyLoss because we already did the logsoftmax in the model
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    #start training
    train_network(greek_train_loader, model, loss_fn, optimizer, epochs)
    #print out the model(for report)
    print(model)
    # start testing
    test_on_new(greek_test_loader, model)

    return

if __name__ == "__main__":
    main(sys.argv)