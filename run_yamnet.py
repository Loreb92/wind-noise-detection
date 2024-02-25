import os
# suppress tensorflow info logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import json
import logging
import argparse
import numpy as np
from scipy import sparse
import pickle
from tqdm import tqdm
from src.utils_audio import read_file_audio
from src.yamnet import YAMNET

# minimum score to keep a prediction. This is useful to keep the output prediction arrays sparse.
MIN_PRED_SCORE = 0.001

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def save_predictions(scores, file_path):
    """ Sparsify and save predictions.
    """

    scores[scores < MIN_PRED_SCORE] = 0
    scores_sp = sparse.coo_array(scores)

    # save predictions
    if file_path.endswith(".npz"):
        sparse.save_npz(file_path, scores_sp)
    elif file_path.endswith(".mtx"):
        mmwrite(file_path, scores_sp)
    else:
        raise ValueError(f"Unknown file format {file_path.split('.')[-1]}.")
    

def save_2D_array(arr, file_path, use_pickle=False):
    """ Save embeddings.
    """
    
    if file_path.endswith(".npz"):
        if use_pickle:
            with open(file_path.replace(".npz", ".pkl"), "wb") as ww:
                pickle.dump(arr, ww)
        else:
            np.savez(file_path, arr)
    elif file_path.endswith(".mtx"):
        mmwrite(file_path, arr)
    else:
        raise ValueError(f"Unknown file format {file_path.split('.')[-1]}.")


def main(args):

    # load model
    logging.info("Loading YAMNet model..")
    yamnet_model = YAMNET()

    if args.classifier_weights_fold:
        # load classifier
        logging.info("Loading classifier model..")
        #from src.feed_forward_nn import WindClassifier
        #classifier_model = WindClassifier(input_dim=yamnet_model._embedding_dim, output_dim=1, activation='sigmoid')
        #classifier_model.load_weights(os.path.join(args.classifier_weights_fold, "model_weights.h5"))
        from tensorflow import keras
        classifier_model = keras.models.load_model(os.path.join(args.classifier_weights_fold, "model"))
        return_embeddings = True
    else:
        return_embeddings = args.save_embeddings


    # save legend of classes
    # TODO:
    # - add option to save this file in custom folder
    with open(os.path.join("data", "class_names.txt"), "w") as f:
        f.write("\n".join(yamnet_model.class_names))

    file_list = os.listdir(args.audio_data_fold)
    logging.info(f"Found {len(file_list)} files in {args.audio_data_fold}.")
    logging.info(f"Running YAMNet predictions..")
    for file in tqdm(file_list, total=len(file_list)):

        if not file.endswith(".wav"):
            continue

        filename_output = file.replace(".wav", f".{args.save_as}")

        # read audio
        waveform, audio_metadata = read_file_audio(os.path.join(args.audio_data_fold, file))

        # predict
        yamnet_output = yamnet_model.predict(waveform, return_embeddings=return_embeddings, return_spectrogram=args.save_spectrogram)

        # save predictions
        scores_file_path = os.path.join(args.output_fold, "scores", filename_output)
        save_predictions(yamnet_output["scores"], scores_file_path)

        # save metadata
        with open(os.path.join(args.output_fold, "metadata", filename_output.replace(f".{args.save_as}", ".json")), "wt") as ww:
            json.dump(audio_metadata, ww)

        if args.save_embeddings:
            embeddings_file_path = os.path.join(args.output_fold, "embeddings", filename_output)
            use_pickle = True if args.save_as == 'npz' else False
            save_2D_array(yamnet_output["embeddings"], embeddings_file_path, use_pickle=use_pickle)

        if args.save_spectrogram:
            spectrogram_file_path = os.path.join(args.output_fold, "spectrograms", filename_output)
            use_pickle = True if args.save_as == 'npz' else False
            save_2D_array(yamnet_output["spectrogram"], spectrogram_file_path, use_pickle=use_pickle)

        if args.classifier_weights_fold:
            classifier_output = classifier_model.predict(yamnet_output["embeddings"], batch_size=32, verbose=0)

            # reshape such that it has same shape as yamnet_output["scores"]
            classifier_output = classifier_output.reshape((-1, 1))

            # save it
            scores_tl_file_path = os.path.join(args.output_fold, "scores_tl", filename_output)
            save_predictions(classifier_output, scores_tl_file_path)

    logging.info("Done.")



if __name__ == "__main__":

    # parse input
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-data-fold', type=str, help="Path to folder containing audio data")
    parser.add_argument('--classifier-weights-fold', type=str, default='none', help='Path to folder containing the weights of the classifier.')
    parser.add_argument('--output-fold', type=str, help="Path to folder where to save output")
    parser.add_argument('--save-embeddings', type=str2bool, default='false', help="Whether to store the embeddings of the audio files")
    parser.add_argument('--save-spectrogram', type=str2bool, default='false', help="Whether to store the spectrogram of the audio files")
    parser.add_argument('--save-as', type=str, default='npz', choices=['npz', 'mtx'], help="The file format to save the outputs. It accepts either npz or mtx (Matrix Market format, which enforces portability). Default is npz.")
    parser.add_argument('--logging-file', type=str, default='logs/run_yamnet.log', help="Path to file where to save the logging")
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

    if args.save_as == 'mtx':
        from scipy.io import mmwrite

    if not os.path.exists(args.output_fold):
        os.mkdir(args.output_fold)
    else:
        raise ValueError(f"Output folder {args.output_fold} already exists.")
    
    # create subfolders for scores, embeddings, and spectrograms
    os.mkdir(os.path.join(args.output_fold, "scores"))
    os.mkdir(os.path.join(args.output_fold, "metadata"))

    if args.save_embeddings:
        os.mkdir(os.path.join(args.output_fold, "embeddings"))
    if args.save_spectrogram:
        os.mkdir(os.path.join(args.output_fold, "spectrograms"))
    if args.classifier_weights_fold == 'none':
        args.classifier_weights_fold = None
    else:
        os.mkdir(os.path.join(args.output_fold, "scores_tl"))
    
    main(args)