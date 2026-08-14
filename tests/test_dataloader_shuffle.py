from mlforge.data import TensorDataset, DataLoader


def test_dataloader_shuffle():

    X = [[i] for i in range(10)]
    y = list(range(10))

    dataset = TensorDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=True,
    )

    batch = next(iter(loader))

    labels = [label for _, label in batch]

    assert len(labels) == 10
    assert sorted(labels) == list(range(10))