# Wind detection YAMNet

This repository contains the scripts used in the paper [Windy events recognition in big bioacoustics datasets using the YAMNet pre-trained Convolutional Neural Network](https://www.sciencedirect.com/science/article/pii/S0048969724050174) to identify segments containing wind noise in audio recordings.

The scripts and data needed to reproduce the results of the paper are provided in the Zenodo repository at <https://zenodo.org/records/11220741>.

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

#### Data

The data has to be organized in the following way:
- `data/annotations_5sec/<file_with_annotations_name>.csv`: it is a CSV file containing the annotations of the audio segments. Each annotation refers to a 5 seconds segment of an audio file.
- `data/<folder_with_audio_files>`: it is a folder containing the audio files in `.wav` format.

The annotation file must be a csv file with the following columns:
- `file_name`: the name of the audio file
- `segment_start_s`: the start time of the annotated segment in seconds belonging to the audio file. The script assumes that the annotation refers to a segment of length 5 seconds.
- `wind`: the label of the annotation. It can be either 0 (absence of wind) or 1 (presence of wind).
The annotation file can contain additional columns.

## Use YAMNet on your own data

### Extract YAMNet predictions

YAMNet was trained on a dataset containing a diverse set of audio events, among which some classes are related to wind. 
This makes it already able to predict windy events in acustic data.

To extract the YAMNet predictions, run the following command:
```
python run_yamnet.py --audio-data-fold data/<folder_with_audio_files> --output-fold data/yamnet_asis_predictions
```
This script reads all the audio files in `.wav` format inside the provided folder and saves the predictions by YAMNet with additional metadata.
It creates two folders inside the output folder:
- `<output_folder>/scores`: contains the predictions in `.npz` format for each audio file. It consists of a matrix with shape `(n_samples, n_classes)` where `n_samples` is the number of audio subsegments in an audio file and `n_classes` is the number of classes in YAMNet. The entries of the matrix corresponds to the probability of the class being present in the subsegment.
- `<output_folder>/metadata`: contains additional information for each audio file.

The script can take other additional arguments:
- `--save-embeddings`: if set to `true`, it saves the embeddings of the audio subsegments in `.npy` format using pickle. The embeddings are the output of the YAMNet model before the final classification layer and can be used to train a new classifier if the target labels are provided.
- `--save-spectrogram`: if set to `true`, it saves the spectrogram of the audio file as a matrix in `.npy` format using pickle.
- `--save-as`: if set to `mtx`, it saves all the output matrices in mtx format.

Please, refer to the notebook `tutorial/1.0.1_read_outputs_of_run_yamnet.ipynb` for more details on how to read the outputs of the script.

### Train YAMNet on an annotated dataset

The last layer of YAMNet before the classification layer provides a vector representation of the audio subsegments that can be used to train a shallow classifier if annotations are available.

Run the following command to create the training dataset using your own annotations and audio data:
```
python make_dataset_from_annotations.py --annotations-file data/annotations_5sec/<file_with_annotations_name>.csv --audio-data-fold data/<folder_with_audio_files> --output-fold <output_folder>
```
This script creates a json file where each row contains the embedding of an audio subsegment and its corresponding annotations. Note that it creates the embedding vectors only of files that are present in the annotation file, not for all the audio files in the folder.

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



## Citation

Terranova, F., Betti, L., Ferrario, V., Friard, O., Ludynia, K., Petersen, G. S., Mathevon, N., Reby, D., & Favaro, L. (2024). Windy events detection in big bioacoustics datasets using a pre-trained Convolutional Neural Network. Science of The Total Environment, 949, 174868. https://doi.org/10.1016/j.scitotenv.2024.174868


```bibtex
@article{TERRANOVA2024174868,
    title = {Windy events detection in big bioacoustics datasets using a pre-trained Convolutional Neural Network},
    journal = {Science of The Total Environment},
    volume = {949},
    pages = {174868},
    year = {2024},
    issn = {0048-9697},
    doi = {https://doi.org/10.1016/j.scitotenv.2024.174868},
    url = {https://www.sciencedirect.com/science/article/pii/S0048969724050174},
    author = {Francesca Terranova and Lorenzo Betti and Valeria Ferrario and Olivier Friard and Katrin Ludynia and Gavin Sean Petersen and Nicolas Mathevon and David Reby and Livio Favaro},
    keywords = {Bioacoustics, Deep learning, Ecoacoustics, Passive Acoustic Monitoring, Soundscape ecology, Wind-noise}
}
```
