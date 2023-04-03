# Recognition-using-Deep-Networks

CS5330 Project 5

This project is about learning how to build, train and test a deep network. We used the MNIST digit dataset to train and test the model. We also modified the network for greek letter recognition. More importantly, we conducted several experiments to show the functionality of each aspect of the model.

## Author

Yao Zhong, zhong.yao@northeastern.edu

## Links

Online report: https://well-turret-89f.notion.site/Project-5-Report-Recognition-using-Deep-Networks-560e4c3827fb4bcf852d2ece68656172

## Environment

MacOS M1 chip

IDE: VS Code

Build with Makefile

## How to run the code

### Before runing

unzip the new_digits and greek_letters zip file, because these images are used in the tasks

### For task 1 A~E (build, train and save the model)

- In the treminal, enter command `python3 model.py` to run, or click the "run" button in the IDE

### For task 1 E~G (test the model)

- In the treminal, enter command `python3 test.py` to run, or click the "run" button in the IDE

### For task 2 (analyse the model)

- In the treminal, enter command `python3 analyse.py` to run, or click the "run" button in the IDE

### For task 3 and extension 2 and extension 3

- In the treminal, enter command `python3 greek.py [mode]` to run.

- "mode" is by default `original`, which will run the task 3

- All options for mode are:
  -- `original` : for task 3, which load the train set provided by the professor
  -- `additional_v1`: for extension 2, which added 3 examples for each letter
  -- `additional_v2`: for extension 2, which added 6 examples for each letter
  -- `extended`: for extension, which extended the greek letters and included omega and delta

### For experiment 1

- In the treminal, enter command `python3 experiment_1.py` to run,or click the "run" button in the IDE

### For experiment 2

- In the treminal, enter command `python3 experiment_2.py` to run,or click the "run" button in the IDE
