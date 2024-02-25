# Wind detection YAMNet

This repository contains the scripts to reproduce the paper "Windy events recognition in big bioacoustics datasets using the YAMNet pre-trained Convolutional Neural Network".

## Setup

#### Environment

The file `environment.yml` contains the dependencies needed to run the scripts.

Using Anaconda, the environment can be installed with the following command:
```
conda env create -f environment.yml
```
then activate the environment with:
```
conda activate wind-noise-detection
```

## Reproduce the results of the paper

The repository contains the 208 audio recordings used for the annotation steps described in the paper _ADD_PATH_FILES_. 
The notebooks `notebook/0.*.ipynb` show how the samples have been chosen, the inter-annotator agreement, and the construction of the final annotated datase.

Run the following command to create the training dataset:
```
python make_dataset_from_annotations.py --annotations-file data/annotations_5sec/annotations_SA_clean.csv --audio-data-fold <fold_with_audio_data> --output-fold <output_folder>
```
This script creates a json file where each row contains the embedding of an audio subsegment and its corresponding annotations.

Then, run the following command to perform the evaluation of the classifier:
```
python eval_yamnet.py --input-dataset-file <dataset_file> --output-file <output_file> --logging-file
```
This script performs an evaluation of YAMNet employing cross-validation.

The subsequent analyses of the paper can be replicated with the Jupyter Notebook `notebook/1.1.0_show_results_model_performance.ipynb`.

The script `eval_yamnet.py` takes an optional argument that can be used to perform the evaluation on a subset of the annotated dataset with half of the size. 
To do that, run:
```
python eval_yamnet.py --input-dataset-file <dataset_file> --output-file <output_file> --logging-file --use-half-data true
```
The subsequent analyses of the paper can be replicated with the Jupyter Notebook `notebook/1.1.1_comparison_performance_half_data.ipynb`.

## Use YAMNet on your own data

### Extract YAMNet predictions

YAMNet was trained on a dataset containing a diverse set of audio events, among which some classes are related to wind. 
This makes it already able to predict windy events in acustic data.

To extract the YAMNet predictions, run the following command:
```
python run_yamnet.py --audio-data-fold <folder_containing_audio_files> --output-fold <output_folder>
```
This script reads all the audio files in `.wav` format inside the provided folder and saves the predictions by YAMNet with additional metadata.
It creates to folders inside the output folder:
- `<output_folder>/scores`: contains the predictions in `.npz` format for each audio file. It consists of a matrix with shape `(n_samples, n_classes)` where `n_samples` is the number of audio subsegments in an audio file and `n_classes` is the number of classes in YAMNet. The entries of the matrix corresponds to the probability of the class being present in the subsegment.
- `<output_folder>/metadata`: contains additional information for each audio file.

The script can take other additional arguments:
- `--save-embeddings`: if set to `true`, it saves the embeddings of the audio subsegments in `.npy` format using pickle. The embeddings are the output of the YAMNet model before the final classification layer and can be used to train a new classifier if the target labels are provided.
- `--save-spectrogram`: if set to `true`, it saves the spectrogram of the audio file as a matrix in `.npy` format using pickle.
- `--save-as`: if set to `mtx`, it saves all the output matrices in mtx format.

### Train YAMNet on an annotated dataset

The last layer of YAMNet before the classification layer provides a vector representation of the audio subsegments that can be used to train a shallow classifier if annotations are available.

Run the following command to create the training dataset using your own annotations and audio data:
```
python make_dataset_from_annotations.py --annotations-file <annotation_file> --audio-data-fold <fold_with_audio_data> --output-fold <output_folder>
```
This script creates a json file where each row contains the embedding of an audio subsegment and its corresponding annotations.
The annotation file must be a csv file with the following columns:
- `file_name`: the name of the audio file
- `segment_start_s`: the start time of the annotated segment in seconds belonging to the audio file. The script assumes that the annotation refers to a segment of length 5 seconds.
Any other column can be used for annotations.
An example of such file is provided in `data/annotations_5sec/FILE`.

Then, run the following command to train a classifier:
```
python train_yamnet.py --input-dataset-file <dataset_file> --model-config-file <config_file> --trained-model-fold <path where to save the model>
```
where the file `config_file` contains the hyperparameters to be used to train the model.
An example of such file is provided in `config/model_train_params.json`.

Finally, to predict the presence of wind run the following command:
```
python run_yamnet.py --audio-data-fold <folder_containing_audio_files> --classifier-weights-fold <path where to save the model> --output-fold <output_folder>
```
In addition to the outputs described in the previous section, this creates a folder `scores_tl` with the predicted scores by the trained model.




