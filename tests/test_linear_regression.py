from mlforge.models.linear_regression import LinearRegression

model = LinearRegression()

X = [1, 2, 3, 4, 5]

predictions = model.forward(X)

print(predictions)

assert len(predictions) == len(X)
assert isinstance(predictions, list)