![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)
![Tensorflow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white)
![Scikit](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)


# RespireNet <img width="60" alt="lungs icon" src="https://github.com/mwritescode/RespireNet/blob/main/assets/imgs/lungs-img.png">
TensorFlow implementation of Microsoft's RespireNet to classify breathing/coughing sounds of people affected by COVID19 using the [Coswara dataset](https://github.com/iiscleap/Coswara-Data). Once the train code is executed, the data is automatically downloaded and pre-precessed, thanks to the tensorflow-datasets pipeline. What you can customize:
* What audios to filter out: you can choose not to consider the `wav` files which last less than k seconds with k between 1 and 4, simply by passing `skip=k` when instantiating a `CoswaraCovidDataset` object
* Whether to use mixup: as a default, for the minority classes a kind of mixup augmentation is produced. Two examples of the same class are randomly selected and concatenated in order to create a new example. This helps in reducing class imbalance, but can be turned off siply passing `mixup=False` when instantiating a `CoswaraCovidDataset`
* What audio file you are interested in: the Coswara dataset is composed of 9 audiofiles per user, but my reduced version only considers four of them (`breathing-deep`, `breathing-shallow`, `cough-heavy` and `cough-shallow`). To select the one you are interested in simply pass `audio_file=file_name` when instantiating a `CoswaraCovidDataset` object
* The final lenght each audio file should have, by default this is set to 7 seconds.
* The low and high cuts for the 5th order Butterworth filetr applied on each data file to reduce noise
* The parameters used for the creation of the Mel Spectrogram using `librosa`

### Requirements
The code requires Python >= 3.6 as well as the following libraries:
* matplotlib
* tensorflow_datasets
* pydub
* python-ffmpeg
* librosa
* scipy
* opencv_python
* tensorflow_gpu
* numpy
* tensorflow

Which can be installed easily through:
```sh
  pip install -U pip
  pip install -r requirements.txt
  ```
  
Finally, in order for the `tfds` audio processing pipeline you also need to have [`ffmpeg`](https://www.ffmpeg.org/) installed on your computer.

### To-Do:
- [ ] Add a proper config file
- [ ] Find a way to filter out the audio files that only contain talking
- [ ] Add option to filter out asymtomatic people, as it doesn't make much sense to try and diagnose them using their cough or breathing
