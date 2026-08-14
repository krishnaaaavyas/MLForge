from abc import ABC, abstractmethod


class Dataset(ABC):
    """
    Base Dataset class.

    Every dataset must implement:

    - __len__()
    - __getitem__()
    """

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass