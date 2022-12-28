from __future__ import absolute_import, division, print_function

from os import mkdir

import tensorflow as tf
from config import EPOCHS, BATCH_SIZE, model_dir
from prepare_data import get_datasets
from models.alexnet import AlexNet
from models.vgg16 import VGG16
from models.vgg19 import VGG19

import numpy as np
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
#from models.proposed_model import proposed_model


def get_model():
    # model = AlexNet()
   # model = proposed_model()
    model = VGG16()

    model.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  metrics=['accuracy'])
    return model
#----------------------------------------------------------------------------
if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    # Check if eager mode on
    print(tf.executing_eagerly())
    print(tf.__version__)

    # Load data
    train_generator, valid_generator, test_generator, \
    train_num, valid_num, test_num = get_datasets()

    #checkpoint_filepath = 'weights.{epoch:02d}-{val_loss:.2f}.h5'
    checkpoint_filepath = 'best_model_Adam_lr3_noAu.h5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)


    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=5, verbose=2, mode='min',
        baseline=None, restore_best_weights=True)

    #checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='best_weights.hdf5', save_best_only=True,
     #                                                save_weights_only=True)


    callback_list = [model_checkpoint_callback]


    model = get_model()
    model.summary()

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # start training
    hist = model.fit(train_generator,
                        epochs=EPOCHS,
                        steps_per_epoch=train_num // BATCH_SIZE,
                        validation_data=valid_generator,
                        validation_steps=valid_num // BATCH_SIZE,
                        callbacks=callback_list)

    np.save('model_history.npy', hist.history)

    # save the whole model
    # Save the entire model as a SavedModel.
    model.save_weights('best_model_Adam_lr3_noAu.h5')
    model.save(model_dir)

