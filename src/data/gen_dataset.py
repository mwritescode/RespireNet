import os

import cv2
from tensorflow.python.ops.gen_data_flow_ops import dynamic_partition
import librosa
import numpy as np

import data.coswara.coswara
import tensorflow as tf
import tensorflow_datasets as tfds

from utils.filters import butter_bandpass_filter

lowcut = 50.0
highcut = 8000.0
fs = 48000
LENGHT = 7 * fs
BATCH_SIZE = 8
PREFETCH_SIZE = 4
n_mels=128
f_min=50
f_max=4000
nfft=2048
hop=512


class CoswaraCovidDataset:
    def __init__(self, 
        audio_file='cough-heavy', 
        split='train', 
        data_dir="../data", 
        pad_with_repeat=True,
        grayscale=False):

        self.split = split
        self.grayscale = grayscale
        self.audio_file = audio_file
        self.pad_with_repeat = pad_with_repeat
        self.data_dir = data_dir

        self.dataset = tfds.load(f'coswara/{audio_file}-skip4-mixup', 
                                split=self.split, 
                                shuffle_files=True,
                                data_dir=data_dir, 
                                as_supervised=True)
       
    def augment_data(self, audio):
        p_roll = tf.random.uniform(shape=[], dtype=tf.dtypes.float32)
        p_pitch_shift = tf.random.uniform(shape=[], dtype=tf.dtypes.float32)
        if p_roll > 0.5:
            audio = tf.roll(audio, int(fs/10), axis=0)
        if p_pitch_shift > 0.3:
            audio = tf.numpy_function(librosa.effects.pitch_shift,
                                        inp=[tf.cast(audio, tf.float32), fs, -2],
                                        Tout=tf.float32)
        
        return audio
    
    def create_melspectrogram(self, audio, grayscale):
        image = librosa.feature.melspectrogram(y=audio.astype(np.float32), 
                                                sr=fs, n_mels=n_mels, 
                                                fmin=f_min, fmax=f_max, 
                                                n_fft=nfft, hop_length=hop)
        image = librosa.power_to_db(image, ref=np.max)
        image = np.nan_to_num((image-image.min()) / (image.max() - image.min()), posinf=0.0, neginf=0.0)
        image *= 255
        if not grayscale:
            image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
            image =  cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
        
    
    def create_features(self, feature, grayscale=False):
        audio, label = feature

        audio = tf.cast(audio, tf.float32)
        audio = tf.numpy_function(butter_bandpass_filter, 
                                inp=[audio, lowcut, highcut, fs], 
                                Tout=tf.double)
        
        if tf.shape(audio)[0] >= LENGHT:
            audio = audio[:LENGHT]
        else:
            mode = 'SYMMETRIC' if self.pad_with_repeat else 'CONSTANT'
            audio = tf.pad(audio, paddings=[[0, LENGHT - tf.shape(audio)[0]]], mode=mode)
        
        if self.split == 'train':
            audio = self.augment_data(audio)
        
        image = tf.numpy_function(self.create_melspectrogram,
                                    inp=[audio, grayscale],
                                    Tout=tf.float32)
        
        image = tf.cast(image, tf.float32) / 255.0

        if grayscale:
            image = tf.expand_dims(image, axis=-1)
        
        # label = 0 if subject is healthy, otherwise it's 1
        label = 0 if label == 0 else 1
        label = tf.one_hot(label, depth=2)

        return image, label
        
    def get_dataset(self):
        data = self.dataset.map(lambda x, y: self.create_features([x,y], self.grayscale), num_parallel_calls=tf.data.AUTOTUNE)#.shuffle(BATCH_SIZE * 10) 
        if self.split == 'train':
            data = data.repeat()
        data = data.batch(BATCH_SIZE, drop_remainder=True) \
                    .prefetch(PREFETCH_SIZE)

        return data
    
        