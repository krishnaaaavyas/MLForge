from mlfs.losses.mse import calculate_mse

predictions = [60, 40, 90]
actuals = [70, 42, 85]

print(calculate_mse(predictions, actuals))