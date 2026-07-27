from mlforge.models.logistic_regression import LogisticRegression

model = LogisticRegression()

X = [1, 2, 3, 4]

print(model.predict_proba(X))

print(model.predict(X))