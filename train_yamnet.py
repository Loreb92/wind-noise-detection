import os
# suppress tensorflow info logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import time
import argparse
import logging
import numpy as np
import pandas as pd
from tensorflow import keras

from src.feed_forward_nn import build_ffnn_model
from src.utils_model_training import read_dataset


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
    

def train_ffnn(train_df, output_dim, learning_rate, batch_size, n_epochs, model_args_dict, class_weighting, trained_model_fold):
    
    # split train and validation (20% of the files in validation set)
    valid_df = train_df.groupby("file_name").sample(frac=0.2, random_state=RNG)
    train_df = train_df.drop(valid_df.index)

    # reset index and shuffle
    train_df = train_df.sample(frac=1, random_state=RNG).reset_index(drop=True)
    valid_df = valid_df.sample(frac=1, random_state=RNG).reset_index(drop=True)

    # get feature and target arrays
    train_df = train_df.copy()
    X_train = np.vstack(train_df.embeddings)
    y_train = train_df.wind.values

    valid_df = valid_df.copy()
    X_valid = np.vstack(valid_df.embeddings)
    y_valid = valid_df.wind.values

    # initialize the model
    model = build_ffnn_model(input_dim=X_train.shape[1], output_dim=output_dim, learning_rate=learning_rate, **model_args_dict)

    # define early stopping
    early_stop_callback = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        restore_best_weights=True
        )
    
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

    # save model
    model.save(os.path.join(trained_model_fold, "model"))

    # predict on the train and validation set
    y_train_scores = model.predict(X_train, verbose=0).flatten().tolist()
    train_df.loc[:, 'y_score'] = y_train_scores
    y_valid_scores = model.predict(X_valid, verbose=0).flatten().tolist()
    valid_df.loc[:, 'y_score'] = y_valid_scores

    train_df.drop(columns=['yamnet_target_score', 'top_sounds_yamnet', 'embeddings'], inplace=True)
    valid_df.drop(columns=['yamnet_target_score', 'top_sounds_yamnet', 'embeddings'], inplace=True)

    train_report = {
            **model_args_dict,
            'learning_rate' : learning_rate,
            'batch_size' : batch_size,
            'n_epochs' : n_epochs,
            'class_weight':class_weighting,
            'hist' : hist.history,
            'train_df' : train_df.to_dict(orient='records'),
            'valid_df' : valid_df.to_dict(orient='records'),
            'time_train' : t1_train - t0_train
        }
    
    # save the train report
    json.dump(train_report, open(os.path.join(trained_model_fold, "train_report.json"), 'w'))
    
    return train_report



def main(args):
    
    # read training set
    logging.info("Reading the dataset..")
    data_df = read_dataset(args.input_dataset_file, use_half_data=False)
    logging.info(f"Number of recordings: {data_df.file_name.nunique()}")
    logging.info(f"Number of data subsegments: {data_df.shape[0]}")
    logging.info(data_df.wind.value_counts(1))
    logging.info(data_df.animal_sound.value_counts(1))

    # read model config
    model_config = json.load(open(args.model_config_file, 'r')) 

    # train the model and save it
    logging.info("Training the model..")
    train_ffnn(data_df, output_dim=1, learning_rate=model_config['learning_rate'], batch_size=model_config['batch_size'], n_epochs=model_config['n_epochs'], model_args_dict=model_config['model_args'], class_weighting=bool(model_config['class_weight']=='True'), trained_model_fold=args.trained_model_fold)

    logging.info("Done.")
    


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dataset-file", type=str, required=True, help="Input dataset file.")
    parser.add_argument("--model-config-file", type=str, required=True, help="File containing the specifics of the model.")
    parser.add_argument("--trained-model-fold", type=str, default='logs/train_yamnet.log', help="Where to save the weights of the trained model.")
    parser.add_argument("--logging-file", type=str, default='logs/train_yamnet.log', help="File to write the logs.")
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

    # init folder to save the trained model
    if not os.path.exists(args.trained_model_fold):
        os.makedirs(args.trained_model_fold)

    main(args)