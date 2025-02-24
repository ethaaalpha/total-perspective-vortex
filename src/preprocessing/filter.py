from abc import ABC, abstractmethod
from mne.io import Raw

class AbstractFilter(ABC):
    @classmethod
    @abstractmethod
    def filter(cls, data: Raw, *kwargs) -> Raw:
        """Return the data filtered (as a copy, original data is untouched)"""
        pass

class CutFilter(AbstractFilter):
    @classmethod
    def filter(cls, data: Raw, start=0, end=30):
        cpy = data.copy()
        return cpy.filter(start, end)