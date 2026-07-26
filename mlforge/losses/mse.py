def calculate_mse(predictions, actuals):
    """
    Calculate Mean Squared Error.

    Parameters:
        predictions (list): Model predictions.
        actuals (list): Ground truth values.

    Returns:
        float: Mean Squared Error.
    """

    if len(predictions) != len(actuals):
        raise ValueError("Lists must have the same length.")

    squared_error_sum = 0

    for predicted, actual in zip(predictions, actuals):

        error = actual - predicted

        squared_error_sum += error ** 2

    return squared_error_sum / len(predictions)