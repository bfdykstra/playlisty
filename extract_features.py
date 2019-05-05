def get_features(segment):
    if len(segment) != 0:
        feature_tuple = (avg_energy(segment), avg_mfcc(segment), zero_crossing_rate(segment), avg_spectral_centroid(segment), avg_spectral_contrast(segment))
        all_features = np.concatenate([feat if type(feat) is np.ndarray else np.array([feat]) for feat in feature_tuple])
        return all_features
    return np.zeros((29,)) # length of feature tuple


def avg_energy(segment):
    if len(segment) != 0:
        energy = librosa.feature.rmse(y=segment, frame_length = len(segment))[0]
        # returns (1,t) array, get first element
        return np.mean(energy)

    
def avg_mfcc(segment, sr=22050, n_mfcc=20):
    '''
    Get the average Mel-frequency cepstral coefficients for a segment
    The very first MFCC, the 0th coefficient, does not convey information relevant to the overall shape of the spectrum. 
    It only conveys a constant offset, i.e. adding a constant value to the entire spectrum. We discard it.
    BE SURE TO NORMALIZE
    
    Parameters:
        segment: numpy array, a time series of audio data
        sr: int, sampling rate, default 22050
        n_mfcc: int, the number of cepstral coefficients to return, default 20.
    Returns:
        numpy array of shape (n_mfcc - 1,)
    '''
    if (len(segment) != 0):
        components = librosa.feature.mfcc(y=segment,sr=sr, n_mfcc=n_mfcc ) # return shape (n_mfcc, # frames)

        return np.mean(components[1:], axis=1)


def zero_crossing_rate(segment):
    '''
    Get average zero crossing rate for a segment. Add a small constant to the signal to negate small amount of noise near silent
    periods.
    
    Parameters:
        segment: numpy array, a time series of audio data
    Returns:
        float, average zero crossing rate for the given segment
    '''
   

    rate_vector = librosa.feature.zero_crossing_rate(segment+ 0.0001, frame_length=len(segment))[0] # returns array with shape (1,x)
    return np.mean(rate_vector)


def avg_spectral_centroid(segment, sr=22050):
    '''
    Indicate at which frequency the energy is centered on. Like a weighted mean, weighting avg frequency by the energy.
    Add small constant to audio signal to discard noise from silence
    Parameters:
        segment: numpy array, a time series of audio data
        sr: int, sampling rate
    Returns:
        float, the average frequency which the energy is centered on.
    '''
    centroid = librosa.feature.spectral_centroid(segment+0.01, sr=sr)[0]
    return np.mean(centroid)


def avg_spectral_contrast(segment, sr=22050, n_bands=6):
    '''
    considers the spectral peak, the spectral valley, and their difference in each frequency subband
    
    columns correspond to a spectral band
    
    average contrast : np.ndarray [shape=(n_bands + 1)]
    each row of spectral contrast values corresponds to a given
    octave-based frequency, take average across bands
    
    '''
    contr = librosa.feature.spectral_contrast(segment, sr=sr, n_bands=n_bands)
    return np.mean(contr, axis=1) # take average across bands

