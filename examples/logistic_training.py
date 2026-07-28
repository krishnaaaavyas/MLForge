from mlforge.models import LogisticRegression
from mlforge.losses import BinaryCrossEntropy
from mlforge.optimizers import GradientDescent

X = [1, 2, 3, 4, 5, 6, 7, 8]
y = [0, 0, 0, 0, 1, 1, 1, 1]

model = LogisticRegression(weight=0.0, bias=0.0)

loss_fn = BinaryCrossEntropy()

optimizer = GradientDescent(learning_rate=0.5)

epochs = 500

losses = []

for epoch in range(epochs):

    predictions = model.forward(X)

    loss = loss_fn.forward(predictions, y)

    losses.append(loss)

    gradients = loss_fn.backward(predictions, y)

    model.backward(X, gradients)

    optimizer.step(model.parameters())

    optimizer.zero_grad(model.parameters())

    if epoch % 50 == 0:
        print(
            f"Epoch {epoch:3d} | Loss = {loss:.4f}"
        )

print()

print(model.forward(X))

print(model.predict(X))