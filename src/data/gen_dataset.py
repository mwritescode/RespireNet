import os

import cv2
import librosa
import numpy as np

import tensorflow as tf
import data.coswara.coswara
import tensorflow_datasets as tfds

from utils.filters import butter_bandpass_filter

lowcut = 50.0
highcut = 8000.0
fs = 48000
LENGHT = 7 * fs
BATCH_SIZE = 16
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
        skip=2,
        mixup=True,
        data_dir="../data", 
        pad_with_repeat=True):

        self.split = split
        self.audio_file = audio_file
        self.pad_with_repeat = pad_with_repeat
        self.data_dir = data_dir
        msg = 'mixup' if mixup else ''

        self.dataset = tfds.load(f'coswara/{audio_file}-skip{skip}-{msg}', 
                                split=self.split, 
                                shuffle_files=True,
                                data_dir=data_dir, 
                                as_supervised=True)
       
    def augment_data(self, audio):
        audio = tf.cast(audio, tf.float32)
        p_roll = tf.random.uniform(shape=[], dtype=tf.dtypes.float32)
        p_pitch_shift = tf.random.uniform(shape=[], dtype=tf.dtypes.float32)
        if p_roll > 0.5:
            audio = tf.roll(audio, int(fs/10), axis=0)
        if p_pitch_shift > 0.3:
            audio = tf.numpy_function(librosa.effects.pitch_shift,
                                        inp=[tf.cast(audio, tf.float32), fs, -2],
                                        Tout=tf.float32)
        return audio
    
    def create_melspectrogram(self, audio):
        image = librosa.feature.melspectrogram(y=audio.astype(np.float32), 
                                                sr=fs, n_mels=n_mels, 
                                                fmin=f_min, fmax=f_max, 
                                                n_fft=nfft, hop_length=hop)
        image = librosa.power_to_db(image, ref=np.max)
        with np.errstate(divide='ignore', invalid='ignore'):
            image = np.nan_to_num((image-image.min()) / (image.max() - image.min()), posinf=0.0, neginf=0.0)
        image *= 255
        return image
        
    
    def create_features(self, feature):
        audio, label = feature

        audio = tf.cast(audio, tf.float32)
        audio = tf.numpy_function(butter_bandpass_filter, 
                                inp=[audio, lowcut, highcut, fs], 
                                Tout=tf.double)
        
        audio, _ = tf.numpy_function(librosa.effects.trim, 
                                    inp=[audio, 20], 
                                    Tout=[tf.double, tf.int64])
        
        if tf.shape(audio)[0] >= LENGHT:
            audio = audio[:LENGHT]
        else:
            diff = LENGHT - tf.shape(audio)[0]
            if self.pad_with_repeat:
                n_repetitions = tf.math.floordiv(LENGHT, tf.shape(audio)[0])
                if n_repetitions > 0:
                    audio = tf.tile(audio, [n_repetitions])
                audio = tf.pad(audio, paddings=[[0, LENGHT - tf.shape(audio)[0]]], mode='SYMMETRIC')
            else:
                audio = tf.pad(audio, paddings=[[0, diff]], mode='CONSTANT')
        
        if self.split == 'train':
            audio = self.augment_data(audio)
        
        image = tf.numpy_function(self.create_melspectrogram,
                                    inp=[audio],
                                    Tout=tf.float32)
        
        image = tf.cast(image, tf.float32) / 255.0

        image = tf.expand_dims(image, axis=-1)
        
        # label = 0 if subject is healthy, otherwise it's 1
        label = 0 if label == 0 else 1
        label = tf.one_hot(label, depth=2)

        return image, label
        
    def get_dataset(self):
        data = self.dataset.map(lambda x, y: self.create_features([x,y]), num_parallel_calls=tf.data.AUTOTUNE).shuffle(BATCH_SIZE * 20) 
        if self.split == 'train':
            data = data.repeat()
        data = data.batch(BATCH_SIZE, drop_remainder=True) \
                    .prefetch(PREFETCH_SIZE)

        return data