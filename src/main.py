import numpy as np

from boolean_function import BooleanFunction

outputs = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]  # length 16 for n=4
# BooLeanHandler = BooleanFunction(3)
bf = BooleanFunction.random(3)
print(bf.is_linearly_separable())
