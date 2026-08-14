from .dataset import Dataset


class TensorDataset(Dataset):

    def __init__(self, x, y):

        if len(x) != len(y):
            raise ValueError(
                "Features and labels must have the same length."
            )

        self.x = x
        self.y = y

    def __len__(self):

        return len(self.x)

    def __getitem__(self, index):

        return self.x[index], self.y[index]