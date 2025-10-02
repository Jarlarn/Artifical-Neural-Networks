import numpy as np


class Data:
    def __init__(self, path):
        self.training_set = np.loadtxt(path, delimiter=",", dtype=np.float64)
        self.x1 = self.training_set[:, 0]
        self.x2 = self.training_set[:, 1]
        self.t = self.training_set[:, 2]


training_data = Data("training_set.csv")
validation_data = Data("validation_set.csv")
# print(training_data.x1)
print(validation_data.x1)
