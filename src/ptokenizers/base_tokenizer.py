from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    @abstractmethod
    def encode(self, dataset):
        """Encode the given text"""
        pass
