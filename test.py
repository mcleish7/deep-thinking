#reading in maze data
import numpy as np
data = np.load("test_shells_maze/data/maze_data_train_9/inputs.npy")
lengths = [d for d in data]
print(len(lengths))

data = np.load("test_shells_maze/data/maze_data_test_33/inputs.npy")
lengths = [d for d in data]
print(len(lengths))