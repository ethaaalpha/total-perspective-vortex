from abc import ABC, abstractmethod
from mne.decoding.csp import CSP

class AbstractReducing(ABC):
    @abstractmethod
    def reduce(self, *args, **kwargs):
        pass

class ScikitlearnCSP(AbstractReducing):
    def reduce(self, *args, **kwargs):
        return super().reduce(*args, **kwargs)