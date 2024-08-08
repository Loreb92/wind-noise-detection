import numpy as np
from scipy import signal
from scipy.io import wavfile
from tensorflow import int16 as int16_tf

import warnings
# ignore a scipy warning when reading .wav file because of additional metadata in the file that scipy cannot parse
warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000
CLIP_THRESHOLD = 0.90


######### Functions to read and preprocess audio files #########
def detect_clipping(waveform, sample_rate, clip_threshold=0.90, resolution_sec=0.5):
    """
    Detects clipping in the waveform.

    TODO:
    - can this be speeded up?

    Parameters:
        waveform (np.array): Array containing the audio data.
        sample_rate (int): Sample rate of the audio.
        clip_threshold (float): Threshold to detect clipping. Default is 0.90.
        resolution_sec (float): Resolution in seconds to detect distinct clipping time spans. Default is 0.5.

    Returns:
        timespans_clip (list of tuples): List of tuples indicating the time spans in seconds where clipping is detected. This looks like [(t_start_clip1, t_end_clip1), (t_start_clip2, t_end_clip2), ...]. If no clipping is detected, it returns an empty list.
    """

    # detect clipping
    indices_clip = np.where(np.abs(waveform) >= clip_threshold)[0]

    # transform indices into time spans
    times_clip = (indices_clip / sample_rate).round(2)

    # group consecutive clip times if within resolution_sec
    timespans_clip = []
    for i, t in enumerate(times_clip):
        if i == 0:
            timespans_clip.append([t, t])
        else:
            if t - timespans_clip[-1][1] <= resolution_sec:
                timespans_clip[-1][1] = t
            else:
                timespans_clip.append([t, t])

    # transform into tuples
    timespans_clip = [(t[0], t[1]) for t in timespans_clip]

    return timespans_clip


def resample_audio(waveform, original_sample_rate, desired_sample_rate):
    """
    Resample waveform with a sample rate.

    If subsampling and multiple, use scipy.signal.decimate, otherwise use scipy.signal.resample.
    """

    if original_sample_rate > desired_sample_rate and original_sample_rate % desired_sample_rate == 0:
        # use decimate
        q = original_sample_rate // desired_sample_rate
        waveform = signal.decimate(waveform, q)
    else:
        # use resample
        desired_length = int(round(float(len(waveform)) /
                            original_sample_rate * desired_sample_rate))
        waveform = signal.resample(waveform, desired_length)

    return waveform


def read_file_audio(file_name, format='wav'):
    """
    Reads a wav file and returns the audio data resampled at 16kHz as an array. Assumes mono audio.

    The preprocessing of the raw file follows: https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/inference.py

    Parameters:
        file_name (string): Path to the wav file.
        format (string): Format of the audio file. Only wav supported.

    Returns:
        waveform (np.array): Array containing the audio data.
        metadata (dict): Contains additional information on the file.
    """

    if format not in ['wav']:
        raise ValueError(f"Format {format} not supported.")

    # TODO:
    # - this gives a warning "WavFileWarning: Chunk (non-data) not understood, skipping it.". How does it mean? Is is possible to manage?
    if format == 'wav':
        sample_rate, audio_arr = wavfile.read(file_name, 'rb')

    # Get duration of audio
    duration = len(audio_arr) / sample_rate

    # normalize to [-1, 1]
    # TODO:
    # - after the resampling, the waveform will take values outside the [-1, 1] range. Is it a problem? Are there other resampling methods that preserve the wavefrom in [-1, 1]?
    audio_arr = audio_arr / int16_tf.max
    audio_arr = audio_arr.astype('float32')

    # detect clipping
    times_clipping = detect_clipping(audio_arr, sample_rate, clip_threshold=CLIP_THRESHOLD)

    # resample to 16 kHz
    if sample_rate != SAMPLE_RATE:
        audio_arr = resample_audio(audio_arr, sample_rate, desired_sample_rate=SAMPLE_RATE)

    metadata = {
        'times_clipping': times_clipping,
        'duration': duration,
    }

    return audio_arr, metadata


######### Functions to convert time to frame indices and viceversa #########

def frame_to_second_start(n_frame, hop_sec):
    """ Given a frame number of the frame, return the start time in seconds. """
    assert type(n_frame) == int and n_frame >= 0, "Frame must be a positive integer."
    return n_frame * hop_sec


def second_to_frames(sec, w_len_sec, hop_sec):
    """ Given a second, return the frame numbers of the frames that contain the second."""

    assert sec >= 0, "Second must be positive."
    assert w_len_sec > 0, "Window length must be positive."
    assert hop_sec > 0, "Hop length must be positive."

    imin = int(np.ceil((sec - w_len_sec) / hop_sec) )
    imax = int(np.floor(sec / hop_sec))
    imax = imax if sec % hop_sec != 0 else imax - 1

    if imin > imax:
        # there is no window that contains the second, this happens if hop_sec > w_len_sec
        return None, None

    return max(0, imin), max(0, imax)


def get_frames_from_interval(start, end, w_len_sec, hop_sec):
    """ Given a start and end time in seconds, return the frames that contain the interval. """

    assert start >= 0, "Start time must be positive."
    assert end >= 0, "End time must be positive."
    assert w_len_sec > 0, "Window length must be positive."
    assert hop_sec > 0, "Hop length must be positive."

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