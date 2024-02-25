import gzip
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import gc


def read_dataset(input_dataset_file, use_half_data, rng=None):

    # read data
    data_df = []
    with gzip.open(input_dataset_file, "rt") as rr:
        for line in tqdm(rr, leave=False, desc="Reading data: "):
            line = json.loads(line)

            # remove the embedding
            embedding = line.pop('embedding')
            line['embeddings'] = np.array(embedding, dtype=np.float32)

            data_df.append(line)

    data_df = pd.DataFrame(data_df)

    if use_half_data:
        # make same experiment, but with half of the recordings sampled at random
        filenames_totake = data_df[['file_name']].drop_duplicates().sample(frac=0.5, random_state=rng).file_name.tolist()

        data_df = data_df[data_df.file_name.isin(filenames_totake)].copy().reset_index(drop=True)
        #data_otherhalf_df = data_df[~data_df.file_name.isin(filenames_totake)].copy().reset_index(drop=True)

    return data_df 


def blockKFold(df, nfolds, block_size_seconds=100, rng=None, return_valid_set=True):
    """ For each recording, split it into blocks and assign blocks randomly to train, validation, and test set
    """

    if rng is None:
        rng = np.random.RandomState(42)

    df = df.copy()

    # split each recording into blocks
    df.loc[:, 'n_block'] = df.segment_start_s // block_size_seconds

    splits = [pd.DataFrame()] * nfolds
    for file_name, rows in df.groupby("file_name"):

        block_subgroups = np.array_split(rows.n_block.unique(), nfolds)

        # check if block_subgroups of same size
        for i in range(len(block_subgroups)):
            block_subgroups[i] = list(block_subgroups[i])

        for n in range(nfolds):
            splits[n] = pd.concat([splits[n], rows[rows.n_block.isin(block_subgroups[n])]])

    for n in range(nfolds):

        test_df = splits[n]
        train_df = df[~df.index.isin(test_df.index)]

        if return_valid_set:
            # make valid set
            valid_df = train_df.groupby("file_name").sample(frac=0.2, random_state=rng)
            train_df = train_df.drop(valid_df.index)

            train_df = train_df.sample(frac=1, random_state=rng).reset_index(drop=True)
            valid_df = valid_df.sample(frac=1, random_state=rng).reset_index(drop=True)
            test_df = test_df.sample(frac=1, random_state=rng).reset_index(drop=True)

            yield train_df, valid_df, test_df
        else:
            train_df = train_df.sample(frac=1, random_state=rng).reset_index(drop=True)
            test_df = test_df.sample(frac=1, random_state=rng).reset_index(drop=True)
            yield train_df, test_df


def reset_tensorflow_keras_backend():
    tf.keras.backend.clear_session()
    #tf.reset_default_graph()
    tf.compat.v1.reset_default_graph()
    _ = gc.collect()