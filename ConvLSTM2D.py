# ConvLSTM2D

import os
import cv2
import imgaug.augmenters as iaa
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import *
from tensorflow.keras.layers import *

with tf.device('/GPU:0'):
    all_data_dir = 'data'
    image_height, image_width = 120, 160
    sequence_length = 8
    X, y = [], []

    image_seq_augmenter = iaa.Sequential([
        iaa.Fliplr(0),
        iaa.Crop(percent=(0, 0.1)),
        iaa.LinearContrast((0.75, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 1.0)),
        iaa.Multiply((0.8, 1.2), per_channel=0.2)
    ])

    for idx, class_name in enumerate(os.listdir(all_data_dir)):
        for image_seq_name in os.listdir(os.path.join(all_data_dir, class_name)):
            image_seq = []
            for frame_name in os.listdir(os.path.join(all_data_dir, class_name, image_seq_name)):
                frame = cv2.imread(os.path.join(all_data_dir, class_name, image_seq_name, frame_name))
                frame = cv2.resize(frame, (image_height, image_width))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_seq.append(frame)
            image_seq_aug = image_seq_augmenter(images=image_seq)
            X.extend([image_seq, image_seq_aug])
            y.extend([idx for i in range(2)])

    X = (np.array(X) / 255.0).astype('float32') # (n_samples, n_frames, height, width, channels)
    y = np.array(y)                             # (n_samples)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)

    early_stopping = tf.keras.callbacks.EarlyStopping(restore_best_weights=True,
                                                      patience=10)

    model = tf.keras.Sequential([
        ConvLSTM2D(8, 3, activation='tanh', input_shape=(sequence_length, image_height, image_width, 3),
                   return_sequences=True, data_format='channels_last', recurrent_dropout=0.3),
        BatchNormalization(),
        MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'),
        TimeDistributed(Dropout(0.3)),
    
        ConvLSTM2D(8, 3, activation='tanh', return_sequences=True,
                   data_format='channels_last', recurrent_dropout=0.3),
        BatchNormalization(),
        MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'),
        TimeDistributed(Dropout(0.3)),
    
        ConvLSTM2D(16, 3, activation='tanh', return_sequences=True,
                   data_format='channels_last', recurrent_dropout=0.3),
        BatchNormalization(),
        MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'),
        TimeDistributed(Dropout(0.3)),
    
        ConvLSTM2D(20, 3, activation='tanh', return_sequences=True,
                   data_format='channels_last', recurrent_dropout=0.3),
        BatchNormalization(),
        MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'),
        TimeDistributed(Dropout(0.3)),
    
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(4),
    ])

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy'],
    )

    model_train_hist = model.fit(
        X_train, y_train,
        shuffle=True,
        batch_size=4,
        epochs=70,
        validation_split=0.2,
        callbacks=[early_stopping],
    )

    model_eval_loss, model_eval_acc = model.evaluate(X_test, y_test)
    date_time_format = '%Y_%m_%d__%H_%M_%S'
    current_date_time_dt = dt.datetime.now()
    current_date_time_str = dt.datetime.strftime(current_date_time_dt, date_time_format)

    model_name = f'model__date_time_{current_date_time_str}__loss_{model_eval_loss}__acc_{model_eval_acc}__hand.h5'
    model.save(model_name)

    df_train_hist = pd.DataFrame(model_train_hist.history)
    df_train_hist.loc[:, ['loss', 'val_loss']].plot()
    plt.show()