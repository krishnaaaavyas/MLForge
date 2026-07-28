from mlforge.activations import Sigmoid

sigmoid = Sigmoid()

print(sigmoid.forward(-100))
print(sigmoid.forward(0))
print(sigmoid.forward(100))