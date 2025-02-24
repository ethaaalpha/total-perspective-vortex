# total-perspective-vortex
This project is about building a pipeline to handle eeg data.

## eeg data
An **electroencephalogram** is a medical record of the electrical brain activity as signals.  
A classic setup of *eeg data* acquisition is by putting electrodes on living being head and then record the cerebral activity.  
These datas can be useful for research.  
In our case we want to determine when does the user do a movemement or think of doing a movement (like opening right hand and thinking of doing it).  

> [!NOTE]
> It will be binary classification A or B.  

### electrical signals
The signal obtained by the *eeg setup* will be a electrical signal that can be represented like that :  

<img src="doc/img/example.png" width=550>

This is the representation of the signal obtained by a unique electrode.  In our case we use 64 distincts electrodes.  
Because the data is a recording, we will need to split the recording every $x$ ms to have an usable data.  

#### filtering
For the moment each signal is still unusable. Like every other ML/Deep learning project, we have to filter our data to keep only the one that is essential for us.

As you know we are actually dealing with signals which is a little bit different than handling classic "exel" like dataset. To imagine how we can filter signal you'll have to look a little on how is composed a signal.  

We use Digital Fourier Transform because our signal isn't continuous but discrete.  


# To look
- What is EEG data ?
    1. Display data with MNE package
- Dimensionality Reduction Algorithm ?
- "real time" data stream
- look for scipy

1. Implementer Preprocessing avec PCA, ICA..
2. Classicification
3. Running


## Links
- [DataSet](https://physionet.org/static/published-projects/eegmmidb/eeg-motor-movementimagery-dataset-1.0.0.zip)
- [MNE Package](https://mne.tools/stable/index.html)
- [eeg data](https://www.youtube.com/watch?v=B9ti7boa9jc)
- [Fourier Transform](https://www.youtube.com/watch?v=spUNpyF58BY)
- [Complex Numbers](https://www.youtube.com/watch?v=M6o5CRYfNxA)
- [Linear Algebra & Vectors (&matrix by the way)](https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [Variance](https://fr.wikipedia.org/wiki/Variance_(math%C3%A9matiques)
- [CoVariance](https://fr.wikipedia.org/wiki/Covariance)
