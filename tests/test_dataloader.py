from mlforge.data import TensorDataset
from mlforge.data import DataLoader


def test_dataloader():

    X = [
        [1],
        [2],
        [3],
        [4],
        [5],
    ]

    y = [
        0,
        1,
        0,
        1,
        0,
    ]

    dataset = TensorDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=2
    )

    batches = list(loader)

    assert len(batches) == 3

    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 1