from mlforge.losses import BinaryCrossEntropy
loss = BinaryCrossEntropy()

predictions = [0.9, 0.2, 0.8]

targets = [1, 0, 1]

print(loss.forward(predictions, targets))

print(loss.backward(predictions, targets))