import os
import json
import argparse
import logging
import gzip
import time
from itertools import product
import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow import keras
from src.feed_forward_nn import build_ffnn_model
from src.utils_model_training import blockKFold, reset_tensorflow_keras_backend, read_dataset

NFOLDS = 5
BLOCK_SIZE_SECONDS = 100

# define models and hyperparam lists
MODEL_LIST = [
    ('yamnet_predictions', None),
    ('ffnn', {'activation':'sigmoid'}),
]

RNG = np.random.RandomState(42)

# Set the seed using keras.utils.set_random_seed. This will set:
# 1) `numpy` seed
# 2) `tensorflow` random seed
# 3) `python` random seed
keras.utils.set_random_seed(812)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

#################### Utility functions for training the feed-forward neural network ####################
def train_and_predict_ffnn(train_df, valid_df, test_df, output_dim, learning_rate, batch_size, n_epochs, model_args_dict, class_weighting=False):
    """ Train a feed-forward neural network and predict on the test set.
    """

    # get feature and target arrays
    train_df = train_df.copy()
    X_train = np.vstack(train_df.embeddings)
    y_train = train_df.wind.values

    valid_df = valid_df.copy()
    X_valid = np.vstack(valid_df.embeddings)
    y_valid = valid_df.wind.values

    # initialize the model
    test_df = test_df.copy()
    model = build_ffnn_model(input_dim=X_train.shape[1], output_dim=output_dim, learning_rate=learning_rate, **model_args_dict)

    # define early stopping
    early_stop_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        restore_best_weights=True
        )

    # get train and validation data
    #X_train, y_train = train_data
    #X_valid, y_valid = valid_data

    if class_weighting:
        cweight_0 = y_train.shape[0] / (2 * (y_train == 0).sum())
        cweight_1 = y_train.shape[0] / (2 * (y_train == 1).sum())
        class_weights = {0:cweight_0, 1:cweight_1}
    else:
        class_weights = None

    # fit the model
    t0_train = time.time()
    hist = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=n_epochs,
                        callbacks=[early_stop_callback],
                        shuffle=True,
                        validation_data=(X_valid,y_valid),
                     verbose=0,
                     class_weight=class_weights
                        )
    t1_train = time.time()

    # predict on the train and validation set
    y_train_scores = model.predict(X_train, verbose=0).flatten().tolist()
    train_df.loc[:, 'y_score'] = y_train_scores
    y_valid_scores = model.predict(X_valid, verbose=0).flatten().tolist()
    valid_df.loc[:, 'y_score'] = y_valid_scores

    # get test set and predict
    X_test = np.vstack(test_df.embeddings)

    y_test_scores = model.predict(X_test, verbose=0).flatten()
    test_df.loc[:, 'y_score'] = y_test_scores

    test_df.drop(columns=['yamnet_target_score', 'top_sounds_yamnet', 'embeddings'], inplace=True)
    train_df.drop(columns=['yamnet_target_score', 'top_sounds_yamnet', 'embeddings'], inplace=True)
    valid_df.drop(columns=['yamnet_target_score', 'top_sounds_yamnet', 'embeddings'], inplace=True)

    line = {
            **model_args_dict,
            'learning_rate' : learning_rate,
            'batch_size' : batch_size,
            'n_epochs' : n_epochs,
            'class_weight':class_weighting,
            'hist' : hist.history,
            'test_df' : test_df.to_dict(orient='records'),
            'train_df' : train_df.to_dict(orient='records'),
            'valid_df' : valid_df.to_dict(orient='records'),
            'time_train' : t1_train - t0_train
        }

    return line


def train_param_grid_ffnn(train_df, valid_df, test_df, model_name, model_args, nfold, results_wrtiter=None):
    """ Train the feed-forward neural network with a grid of hyperparameters.
    """

    # define hparams for the ffnn
    LEARNING_RATES = [5e-5, 2e-4, 5e-4]
    N_EPOCHS = [20]
    BATCH_SIZES = [64]
    CLASS_WEIGHTS = [True, False]

    for hparms in tqdm(product(LEARNING_RATES, BATCH_SIZES, N_EPOCHS, CLASS_WEIGHTS), 
                       total=len(LEARNING_RATES)*len(BATCH_SIZES)*len(N_EPOCHS)*len(CLASS_WEIGHTS), 
                       leave=False, desc="Hyperparam loop: "):

        lr, batch_size, n_epochs, class_w = hparms

        t0_train_eval = time.time()
        fit_report_line = train_and_predict_ffnn(train_df=train_df,
                                                 valid_df=valid_df,
                                                 test_df=test_df,
                                                 output_dim=1,
                                                 learning_rate=lr,
                                                 batch_size=batch_size,
                                                 n_epochs=n_epochs,
                                                 model_args_dict=model_args,
                                                 class_weighting=class_w)
        t1_train_eval = time.time()
        fit_report_line['model_name'] = model_name
        fit_report_line['time_train_eval'] = t1_train_eval - t0_train_eval
        fit_report_line['n_fold'] = nfold

        if results_wrtiter is not None:
            # write results on file
            results_wrtiter.write(json.dumps(fit_report_line) + "\n")

        reset_tensorflow_keras_backend()


#################### Utility functions for YAMNet as-is ####################
def save_predictions_yamnet_asis(train_df, valid_df, test_df, model_name, model_args, nfold, results_wrtiter=None):
    """ Train YAMNet as-is.
    """

    # get predictions of all splits, those are already present in the input dataset
    train_df = train_df.copy()
    train_df.drop(columns=['top_sounds_yamnet', 'embeddings'], inplace=True)
    train_df = train_df.rename(columns={'yamnet_target_score':'y_score'})

    valid_df = valid_df.copy()
    valid_df.drop(columns=['top_sounds_yamnet', 'embeddings'], inplace=True)
    valid_df = valid_df.rename(columns={'yamnet_target_score':'y_score'})

    test_df_ = test_df.copy()
    test_df_.drop(columns=['top_sounds_yamnet', 'embeddings'], inplace=True)
    test_df_ = test_df_.rename(columns={'yamnet_target_score':'y_score'})

    fit_report_line = {
                    'model_name' : model_name,
                    'n_fold' : nfold,
                    'test_df' : test_df_.to_dict(orient='records'),
                    'train_df' : train_df.to_dict(orient='records'),
                    'valid_df' : valid_df.to_dict(orient='records')
                    }
    
    if results_wrtiter is not None:
        # write results on file
        results_wrtiter.write(json.dumps(fit_report_line) + "\n")


def main(args):
    
    logging.info("Reading the dataset..")
    if args.use_half_data:
        logging.info("Using half of the data..")
    data_df = read_dataset(args.input_dataset_file, args.use_half_data, rng=RNG)
    logging.info(f"Number of recordings: {data_df.file_name.nunique()}")
    logging.info(f"Number of data subsegments: {data_df.shape[0]}")
    logging.info(data_df.wind.value_counts(1))
    logging.info(data_df.animal_sound.value_counts(1))

    logging.info("Start training..")

    # open file to save results
    fopen_results = gzip.open(args.output_file, "wt")

    # define cross-validation splits
    crossvalsplits = blockKFold(data_df, nfolds=NFOLDS, block_size_seconds=BLOCK_SIZE_SECONDS, rng=RNG)

    # look across all cross-validation splits
    for nfold, (train_df, valid_df, test_df) in tqdm(enumerate(crossvalsplits), total=NFOLDS, desc="Inner CV loop: "):

        # loop across all models
        for model_name, model_args in tqdm(MODEL_LIST, total=len(MODEL_LIST), leave=False, desc="Training and evaluating models: "):

            if model_name == 'yamnet_predictions':
                # use YAMNet as-is
                save_predictions_yamnet_asis(train_df=train_df, valid_df=valid_df, test_df=test_df, model_name=model_name, model_args=model_args, nfold=nfold, results_wrtiter=fopen_results)
            elif model_name == 'ffnn':
                # train a feed-forward neural network
                #train_param_grid_ffnn(train_data=(X_train, y_train), valid_data=(X_valid, y_valid), test_df=test_df, model_name=model_name, model_args=model_args, nfold=nfold, results_wrtiter=fopen_results)
                train_param_grid_ffnn(train_df=train_df, valid_df=valid_df, test_df=test_df, model_name=model_name, model_args=model_args, nfold=nfold, results_wrtiter=fopen_results)

            else:
                logging.info(f"Model {model_name} not recognized, skipping.")

    fopen_results.close()

    logging.info("Evaluation ends!")


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dataset-file", type=str, required=True, help="Input dataset file.")
    parser.add_argument("--output-file", type=str, required=True, help="File to write the results.")
    parser.add_argument("--logging-file", type=str, default='logs/eval_yamnet.log', help="File to write the logs.")
    parser.add_argument("--use-half-data", type=str2bool, default='false', help="If true, use half of the data.")
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

    # initialize results folder
    res_fold = os.path.dirname(args.output_file)
    if not os.path.exists(res_fold):
        os.makedirs(res_fold)

    main(args)