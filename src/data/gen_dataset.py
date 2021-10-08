import os

import cv2
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
PREFETCH_SIZE = 2
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
        use_mixup=False,
        use_concat=True,
        grayscale=False):

        self.split = split
        self.grayscale = grayscale
        self.audio_file = audio_file
        self.data_dir = data_dir

        self.dataset = tfds.load(f'coswara/{audio_file}', 
                                split=self.split, 
                                shuffle_files=True,
                                data_dir=data_dir, 
                                as_supervised=True)

        self.dataset = self.dataset.filter(self.filter_by_lenght) \
                                    .map(self.preprocess_data, num_parallel_calls=tf.data.AUTOTUNE) \
                                    .map(lambda x, y : self.apply_padding([x,y], pad_with_repeat),
                                                    num_parallel_calls=tf.data.AUTOTUNE)

        if (use_mixup or use_concat) and self.split == 'train':
            positive_data = self.dataset.filter(self.filter_by_class)
            pos_1 = positive_data.shuffle(BATCH_SIZE).batch(BATCH_SIZE)
            pos_2 = positive_data.shuffle(BATCH_SIZE).batch(BATCH_SIZE)
            positive_data = tf.data.Dataset.zip((pos_1, pos_2))
            if use_mixup:
                positive_data = positive_data.map(
                    lambda ds_1, ds_2: self.mixup(ds_1, ds_2, alpha=0.3), num_parallel_calls=tf.data.AUTOTUNE)
            else:
                positive_data = positive_data.map(
                    lambda ds_1, ds_2: self.concat_aug(ds_1, ds_2), num_parallel_calls=tf.data.AUTOTUNE)
            self.dataset = self.dataset.concatenate(positive_data.unbatch())

    def preprocess_data(self, audio, label):
        audio = tf.numpy_function(butter_bandpass_filter, 
                                    inp=[audio, lowcut, highcut, fs], 
                                    Tout=tf.double)
        max_val = tf.reduce_max(audio)
        audio, _ = tf.numpy_function(librosa.effects.trim, 
                                    inp=[audio[int(fs):], 20, max_val, 256, 64], 
                                    Tout=[tf.double, tf.int64])

        # label 0: negative, label 1: positive
        label_id = 0 if label == 0 else 1

        return audio, label

    def repeat_until_len(self, audiofile):
        audio_size = len(audiofile)
        if audio_size >= LENGHT:
            result = audiofile[:LENGHT]
        else:
            result = np.zeros(shape=(LENGHT,), dtype='float32')
            idx = 0
            while idx + audio_size <= LENGHT:
                result[idx:idx+audio_size] = audiofile
                idx += 1
            if idx < LENGHT:
                result[idx:LENGHT] = audiofile[:LENGHT-idx]
        return result

    def zero_padding(self, audiofile):
        audio_size = len(audiofile)
        if audio_size > LENGHT:
            result = audiofile[:LENGHT]
        else:
            zero_pad = np.zeros(LENGHT-audio_size)
            result = np.concatenate([audiofile, zero_pad])
        return tf.cast(result, tf.float32)

    def apply_padding(self, feature, pad_with_repeat):
        audio, label = feature
        audio = tf.cast(audio, tf.float32)
        padding_func = self.repeat_until_len if pad_with_repeat else self.zero_padding
        audio = tf.numpy_function(padding_func,
                                  inp=[audio], 
                                  Tout=tf.float32)
        
        return audio, label


    def filter_by_class(self, audio, label):
        result = tf.reshape(tf.math.greater(label, 0), [])
        return result

    def filter_by_lenght(self, audio, label):
        # Filter out audios that last less than 1 second
        duration = tf.numpy_function(librosa.get_duration, 
                                    inp=[tf.cast(audio, tf.float32), fs], 
                                    Tout=tf.double)
        result = tf.reshape(tf.math.greater_equal(duration, 2), [])
        return result

    def sample_beta_distribution(self, size, concentration_0=0.2, concentration_1=0.2):
        gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
        gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
        return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

    def mixup(self, ds_1, ds_2, alpha=0.2):
        audio_1, label_1 = ds_1
        audio_2, _ = ds_2
        batch_size = BATCH_SIZE
        
        l = self.sample_beta_distribution(batch_size, alpha, alpha)
        x_l = tf.reshape(l, (batch_size, 1))
        
        audios = audio_1 * x_l + audio_2 * (1 - x_l)
        labels = label_1
        
        return audios, labels

    def concat_aug(self, ds_1, ds_2, lenght=LENGHT):
        audio_1, label_1 = ds_1
        audio_2, _ = ds_2
        if lenght % 2 == 0:
            half_1 = audio_1[..., :int(lenght/2)]
            half_2 = audio_2[..., int(lenght/2):]
        else:
            half_1 = audio_1[..., :int(lenght+1/2)]
            half_2 = audio_2[..., int(lenght-1/2):]
            
        audios = tf.concat([half_1, half_2], axis=-1)
        labels = label_1
        
        return audios, labels
    
    def augment_data(self, audio):
        p_add_noise = tf.random.uniform(shape=[], dtype=tf.dtypes.float32)
        p_roll = tf.random.uniform(shape=[], dtype=tf.dtypes.float32)
        p_pitch_shift = tf.random.uniform(shape=[], dtype=tf.dtypes.float32)
        #if p_add_noise > 0.4:
        #    audio = 0.002*tf.random.normal(shape=tf.shape(audio), mean=0, stddev=1)
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
        image = np.nan_to_num((image-image.min()), (image.max() - image.min()))
        image *= 255
        if not grayscale:
            image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
            image =  cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
        
    
    def create_features(self, feature, grayscale=False):
        audio, label = feature
        
        if self.split == 'train':
            audio = self.augment_data(audio)
        
        image = tf.numpy_function(self.create_melspectrogram,
                                    inp=[audio, grayscale],
                                    Tout=tf.float32)
        
        image = tf.cast(image, tf.float32) / 255.0

        if grayscale:
            image = tf.expand_dims(image, axis=-1)
        label = tf.one_hot(label, depth=2)
        return image, label
        
    def get_dataset(self):
        data = self.dataset.map(lambda x, y: self.create_features([x,y], self.grayscale), num_parallel_calls=tf.data.AUTOTUNE)#.shuffle(BATCH_SIZE * 10) 
        if self.split == 'train':
            data = data.repeat()
        data = data.batch(BATCH_SIZE, drop_remainder=True) \
                    .prefetch(PREFETCH_SIZE)

        return data
    
        