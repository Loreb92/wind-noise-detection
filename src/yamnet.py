import csv
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Utility functions to extract YAMNET output corresponding to a time interval
def frame_to_second_start(frame, sr=16000, w_len_sec=0.96, hop_sec=0.48):
    """ Given a frame number of the frame, return the start time in seconds. """
    assert type(frame) == int and frame >= 0, "Frame must be a positive integer."
    return frame * hop_sec


def second_to_frames(sec, w_len_sec=0.96, hop_sec=0.48):
    """ Given a second, return the frame numbers of the frames that contain the second."""

    assert sec >= 0, "Second must be positive."

    imin = int(np.ceil((sec - w_len_sec) / hop_sec) )
    imax = int(np.floor(sec / hop_sec))
    imax = imax if sec % hop_sec != 0 else imax - 1

    if imin > imax:
        # there is no window that contains the second, this happens if hop_sec > w_len_sec
        return None, None

    return max(0, imin), max(0, imax)


def get_frames_from_interval(start, end, w_len_sec=0.96, hop_sec=0.48):
    """ Given a start and end time in seconds, return the frames that contain the interval. """

    # get the frames that contain the start and end times
    start_frames = second_to_frames(start, w_len_sec=w_len_sec, hop_sec=hop_sec)
    end_frames = second_to_frames(end, w_len_sec=w_len_sec, hop_sec=hop_sec)

    if start_frames[0] is None or end_frames[0] is None:
        return []

    # get the frames that contain the interval
    frames = sorted(list(set(start_frames + end_frames)))

    if len(frames) == 1:
        frames = [frames[0]]
    else:
        frames = list(range(frames[0], frames[-1]+1))

    return frames


class YAMNET:
    """
    Class that wraps the YAMNet model.
    """
    def __init__(self, model_path=None):
        """
        Parameters:
            model_path (string): Path to the YAMNet model. If None, the model will be downloaded.
        """
        # TODO: 
        # - allow to load yamnet from a local file (how to do that??)
        # see this: https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_visualization.ipynb
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        self.class_names = self._get_class_names()
        self._w_len_sec = 0.96
        self._w_hop_sec = 0.48
        self._sample_rate = 16000
        self._embedding_dim = 1024


    def _get_class_names(self):
        """
        Read the class name definition file and return a list of strings.
        From: https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet.py
        """
        class_map_csv = self.model.class_map_path().numpy()
        with open(class_map_csv) as csv_file:
            reader = csv.reader(csv_file)
            next(reader)   # Skip header
            return [display_name for (_, _, display_name) in reader]
        

    def predict(self, waveform, return_embeddings=False, return_spectrogram=False):
        """
        Predict the classes of an audio waveform.

        Parameters:
            waveform (1D np.array): Array containing the audio data.
            return_embeddings (bool): If True, return also the embeddings.
            return_spectrogram (bool): If True, return also the spectrogram.

        Returns:
            scores (dict or np.array): Array containing the scores for each class and for each frame. If return_embeddings and/or return_spectrogram are True, return a dictionary with the scores, the embeddings, and/or spectrogram. 
        """
        scores, embeddings, spectrogram = self.model(waveform)

        output = {'scores': scores.numpy()}
        if return_embeddings:
            output['embeddings'] = embeddings.numpy()
        if return_spectrogram:
            output['spectrogram'] = spectrogram.numpy()

        return output