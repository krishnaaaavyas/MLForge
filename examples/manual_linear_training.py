from mlforge.models import LinearRegression
from mlforge.losses import MeanSquaredError
from mlforge.optimizers import GradientDescent

X = [1, 2, 3, 4]
y = [3, 5, 7, 9]

model = LinearRegression()

loss_fn = MeanSquaredError()

optimizer = GradientDescent(learning_rate=0.01)

epochs = 100

for epoch in range(epochs):

    predictions = model.forward(X)

    loss = loss_fn.forward(predictions, y)

    gradients = loss_fn.backward(predictions, y)

    model.backward(X, gradients)

    optimizer.step(model.parameters())

    optimizer.zero_grad(model.parameters())

    if epoch % 10 == 0:
        print(epoch, loss)

print(model.forward(X))