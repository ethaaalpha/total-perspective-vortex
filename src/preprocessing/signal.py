from abc import ABC, abstractmethod
from mne.io import Raw
from matplotlib import pyplot as pp
import numpy as np

class AbstractSignalAnalyser(ABC):
    def __init__(self, raw: Raw):
        self.raw = raw
        self.data = np.array(raw.get_data())
        # number of points per seconds (Hz)
        self.sampling_frequency = raw.info["sfreq"]

    @abstractmethod
    def analyse(self):
        """Analyse the passed raw mne data"""
        pass

    def show(self):
        pp.show()

class FastFourierTransform(AbstractSignalAnalyser):

    def __init__(self, raw, name="FastFourierTransform"):
        self.name = name
        super().__init__(raw)

    def analyse(self):
        pp.figure()
        pp.xlabel("Frequency (Hz)")
        pp.ylabel("Magnitude")
        pp.xlim(0, 40)
        pp.ylim(0, 0.35)
        pp.title(self.name)
        pp.gcf().canvas.manager.set_window_title("Signal Analysis")
        pp.grid()

        for signal in self.data:
            N = len(signal)
            fourier_transform = np.fft.fft(signal)
            fourier_frequencies = np.fft.fftfreq(N, 1 / self.sampling_frequency)

            # we use abs to compute the imaginary part and real part (cf: magnitude of complex numbers)
            # we do use :N // 2 to only keep positive frequencies due to complex values symmetry
            pp.plot(fourier_frequencies[:N // 2], np.abs(fourier_transform[:N // 2]))
