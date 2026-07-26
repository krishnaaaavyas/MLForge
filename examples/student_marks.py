from mlforge.models.linear_regression import LinearRegression
from mlforge.training.trainer import Trainer
from mlforge.optimizers import GradientDescent

X = [1,2,3,4,5]

y = [13,21,29,37,45]

model = LinearRegression()

optimizer = GradientDescent(
    learning_rate=0.01
)

trainer = Trainer(optimizer)

trainer.fit(
    model,
    X,
    y,
    epochs=200,
)

print()

print(model.predict(X))