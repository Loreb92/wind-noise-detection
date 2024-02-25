import os
# suppress tensorflow info logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import json
import gzip
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.utils_audio import read_file_audio, get_frames_from_interval
from src.yamnet import YAMNET

# set here the duration of the annotated segments
ANNOTATED_SEGMENT_DURATION_SEC = 5

# minimum time of clipping to consider the segment as clipped
MIN_TIME_CLIPPING_SEC = 0.5

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def find_overlapping_elements(lst, interval):
    result = []
    for sub_list in lst:
        # Assuming time-ordered lists
        if sub_list[0] > interval[1]:
            break  # No need to check further as sublist starts after the interval ends
        if interval[0] < sub_list[1] and interval[1] > sub_list[0]:
            result.append(sub_list)
    return result


def create_annotated_dataset(annotations_df, model, audio_data_fold, output_fold, save_spectrogram=False):

    # get the parameters of the model referring to the segmentation of the audio recording
    W_HOP_SEC = model._w_hop_sec
    W_LEN_SEC = model._w_len_sec

    # get the class names
    class_to_idx = {v : k for k, v in enumerate(model.class_names)}
    idx_to_class = {v:k for k, v in class_to_idx.items()}

    # select the classes of interest
    target_class_names = ["Wind", "Wind noise (microphone)", "Rustling leaves"]
    idx_target_classes = [class_to_idx[cn] for cn in target_class_names ]

    # open output file
    output_file = os.path.join(output_fold, "dataset_annotated.json.gz")
    fopen = gzip.open(output_file, "wt")

    for file_name in tqdm(annotations_df.file_name.unique(), total=annotations_df.file_name.nunique()):

        audio_file = os.path.join(audio_data_fold, file_name)
        if not os.path.exists(audio_file):
            logging.info(f"Audio file with annotations {audio_file} does not exist, skipped.")
            continue

        # get annotations for this file
        file_annotations = annotations_df[annotations_df.file_name == file_name]

        # read audio file
        waveform, audio_metadata = read_file_audio(audio_file)

        # predict
        output = model.predict(waveform, return_embeddings=True, return_spectrogram=save_spectrogram)

        for i in range(file_annotations.shape[0]):
            row_annotation = file_annotations.iloc[i]
            segment_start_s = row_annotation.segment_start_s

            # Take all embeddings referring to times from segment_start_s to segment_start_s + ANNOTATED_SEGMENT_DURATION_SEC
            frame_idxs = get_frames_from_interval(segment_start_s, segment_start_s + ANNOTATED_SEGMENT_DURATION_SEC, W_LEN_SEC, W_HOP_SEC)

            # get the embeddings laying fully in the annotated interval
            frame_idxs = [i for i in frame_idxs if (i * W_HOP_SEC) >= segment_start_s and (i * W_HOP_SEC + W_LEN_SEC) <= segment_start_s + ANNOTATED_SEGMENT_DURATION_SEC ]

            # get the corresponding embeddings and yamnet scores
            embeddings_frame = output['embeddings'][frame_idxs]
            scores_yamnet_frame = output['scores'][frame_idxs]#.mean(axis=0)
            scores_yamnet_frame = scores_yamnet_frame[:, idx_target_classes].max(axis=1)
            #scores_yamnet_target_class_frame = scores_yamnet_frame[idx_target_classes].max()

            # get the top 5 predictions of Yamnet
            top_sounds_with_score = [[idx_to_class[i], round(scores_yamnet_frame[i].item(), 4)] for i in np.argsort(scores_yamnet_frame)[::-1][:5]]

            # copy the labels of the segment to all the embeddings
            clippings = find_overlapping_elements(audio_metadata['times_clipping'], [segment_start_s, segment_start_s + ANNOTATED_SEGMENT_DURATION_SEC])
            total_clipping_time = sum([clip[1] - clip[0] for clip in clippings])
            for i in range(embeddings_frame.shape[0]):
                embedding_subseg = embeddings_frame[i].tolist()
                yamnet_target_score_subseg = scores_yamnet_frame[i]
                line = {
                    'file_name' : file_name,
                    'segment_start_s' : segment_start_s,
                    'n_subsegment' : i,
                    **row_annotation.drop(['file_name', 'segment_start_s']).to_dict(),
                    'clip' : int(total_clipping_time > MIN_TIME_CLIPPING_SEC),
                    'yamnet_target_score' : round(yamnet_target_score_subseg.item(), 4),
                    'top_sounds_yamnet' : top_sounds_with_score,
                    'embedding':embedding_subseg
                }
                line_string = json.dumps(line)

                fopen.write(line_string + "\n")

    fopen.close()


def main(args):

    # load model
    model = YAMNET()

    # read annotation data
    # it must have columns 'file_name', referring to the audio recording file, and 'segment_start_s', referring to the second in which the annotated audio segment starts
    # TODO: include different file formats
    annotations_df = pd.read_csv(args.annotations_file)
    logging.info(f"Loaded {annotations_df.shape[0]} annotations from {annotations_df.file_name.nunique()} audio files.")

    if annotations_df[['file_name', 'segment_start_s']].duplicated().any():
        raise ValueError("There are duplicated identifiers in the annotations file.")
    
    logging.info(f"Creating annotated dataset..")
    create_annotated_dataset(annotations_df, model, args.audio_data_fold, args.output_fold, args.save_spectrogram)
    logging.info(f"Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-file", type=str, help="Path to annotations file")
    parser.add_argument("--audio-data-fold", type=str, help="Path to folder containing audio data, assuming it is a csv file")
    parser.add_argument("--output-fold", type=str, help="Path to folder where to save output")
    parser.add_argument("--save-spectrogram", type=str2bool, default='false', help="Whether to store the spectrogram of the audio files")
    parser.add_argument("--logging-file", type=str, default='logs/make_dataset_from_annotations.log', help="Path to file where to save the logging")
    args = parser.parse_args()

    # initialize logger
    log_dir = os.path.dirname(args.logging_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(args.logging_file)
    if os.path.exists(log_file):
        raise FileExistsError('The log file already exists. Likely, this script has already run with the input configuration.')

    # set the level and format of the logging    
    logging.basicConfig(filename=log_file, level=logging.INFO, format='[%(asctime)s] [%(levelname)s]: %(message)s ', datefmt='%Y-%m-%d %H:%M:%S')

    # make the logger print on terminal
    logging.getLogger().addHandler(logging.StreamHandler())

    if not os.path.exists(args.output_fold):
        os.mkdir(args.output_fold)
    else:
        raise ValueError(f"Output folder {args.output_fold} already exists.")
    
    if args.save_spectrogram:
        os.mkdir(os.path.join(args.output_fold, "spectrograms"))

    main(args)