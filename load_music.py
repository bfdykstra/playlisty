import librosa


def get_song_names(directory):
    '''
    Get a list of song file paths from a directory
    Parameters:
        directory: string, the directory where you are looking for audio files
    Returns:
        list comprised of string file paths
    '''
    
    # librosa has utility methods that find readable file formats given a directory
    return librosa.util.find_files(directory)


def load_song(file_path):
    '''
    load an audio file given a file path
    Parameters:
        file_path: string, valid file path
    Returns:
        tuple, (time series of audio file, sampling rate)
    '''
    try:
        signal, sr = librosa.load(file_path)
        return (signal, sr)
    except Exception as e:
        print "Error loading {}: {}".format(file_path, e)
        raise e