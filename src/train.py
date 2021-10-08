import tensorflow as tf
from tensorflow.python.keras.backend import binary_crossentropy

from models.respirenet import respirenet
from data.gen_dataset import CoswaraCovidDataset

# TODO: find a way to evaluate also precision, recall and f1 score
# TODO: write train notebook displaying history and confusion matrix

model = respirenet(input_shape=(128, 657, 1))
ds_train = CoswaraCovidDataset(grayscale=True, use_concat=False).get_dataset()
ds_val = CoswaraCovidDataset(grayscale=True, split='validation', use_concat=False).get_dataset()
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(ds_train, validation_data=ds_val, epochs=100, steps_per_epoch=10)