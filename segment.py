import numpy as np
import librosa

"""
    Segment a signal using onset detection
    returns:
        dictionary with attributes segemented and shape
        
"""
def segment_onset(signal, sr, hop_length, backtrack=True):
    # Compute the frame indices for estimated onsets in a signal
    onset_samples = librosa.onset.onset_detect(signal, sr=sr, hop_length=hop_length, backtrack=backtrack, units='samples')
    
    # return np array of audio segments, within each segment is the actual audio data
    prev_ndx = 0
    segmented = []
    for frame in onset_samples:
        segmented.append(np.array(signal[prev_ndx:frame]))
        prev_ndx = frame
    return {'segmented': np.array(segmented), 'shape': np.array(segmented).shape}
    
# return a np array of shape (num_segments, # of hops that fit in entire signal)
def segment(signal, num_segments=1024, hop_length=512):
    frames = librosa.util.frame(signal, frame_length=num_segments, hop_length=hop_length)
    
    return frames

    