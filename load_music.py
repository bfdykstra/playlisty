import librosa


def get_song_names(directory):
    # librosa has utility methods that find readable file formats given a directory
    return librosa.util.find_files(directory)

def load_song(file_path):
    try:
        signal, sr = librosa.load(file_path)
        return (signal, sr)
    except Exception as e:
        print "Error loading {}: {}".format(file_path, e) 