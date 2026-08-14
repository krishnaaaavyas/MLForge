from math import ceil
import random

class DataLoader:

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
    ):

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):

        return ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):

        self.indices = list(range(len(self.dataset)))

        if self.shuffle:
            random.shuffle(self.indices)

        self.index = 0

        return self
    
    def __next__(self):

        if self.index >= len(self.dataset):
            raise StopIteration

        batch = []

        end = min(
            self.index + self.batch_size,
            len(self.dataset)
        )

        for i in range(self.index, end):
            sample_index = self.indices[i]
            batch.append(self.dataset[sample_index])

        self.index = end

        return batch