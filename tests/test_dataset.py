from mlforge.data.tensor_dataset import TensorDataset


def test_tensor_dataset():

    X = [
        [1, 2],
        [3, 4],
        [5, 6],
    ]

    y = [0, 1, 0]

    dataset = TensorDataset(X, y)

    assert len(dataset) == 3

    sample, label = dataset[1]

    assert sample == [3, 4]
    assert label == 1