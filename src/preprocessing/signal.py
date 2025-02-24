from abc import ABC, abstractmethod
from mne.io import Raw

class AbstractSignalAnalyser(ABC):
    def analyse(raw: Raw):
        """Analyse the passed raw mne data"""
        pass

    @abstractmethod
    def extract_frequencies(signal):
        """This function is supposed to return the list of frequencies that are inside the passed signal"""
        pass

class FourierTransform(AbstractSignalAnalyser):
    def analyse(raw):
        return super().analyse()
    
    def extract_frequencies(signal):
        return super().extract_frequencies()