import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Reshape, UpSampling1D
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import argparse
import os


def main(input_file, model_file):
    assert os.path.exists(input_file)
    print("Loading data")
    data = pd.read_csv(input_file, header=None)
    data = data.values
    sdata = data.shape
    data = np.reshape(data, [sdata[0], sdata[1], 1])
    print("Loaded data")

    # Autoencoder parameters
    inputDim = sdata[1]
    latentDim = 64
    filters = 128
    dictionarySize = 22
    lr = 1e-4
    epochs = 50
    batch_size = 512
    validation_split = 0.1

    # Encoder
    x0 = Input(shape=(inputDim, 1), name='encoder_input')
    x1 = Conv1D(filters, 7, padding='same', activation='relu', name='conv1')(x0)
    x2 = MaxPooling1D(2, name='max_pool1')(x1)
    x3 = Conv1D(filters, 5, padding='same', activation='relu', name='conv2')(x2)
    x4 = MaxPooling1D(2, name='max_pool2')(x3)
    sx4 = K.int_shape(x4)
    x5 = Flatten()(x4)
    sx5 = K.int_shape(x5)
    latent = Dense(latentDim, name='latent_vector')(x5)

    # Decoder
    y1 = Dense(sx5[1])(latent)
    y2 = Reshape((sx4[1], sx4[2]))(y1)
    y3 = UpSampling1D(2, name='up_sample1')(y2)
    y4 = Conv1D(filters, 5, padding='same', activation='relu', name='decon1')(y3)
    y5 = UpSampling1D(2, name='up_sample2')(y4)
    output = Conv1D(dictionarySize, 7, padding='same', activation='softmax', name='decon2')(y5)

    # Autoencoder
    autoencoder = Model(x0, output, name='autoencoder')
    autoencoder.summary()
    autoencoder.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr))
    hist = autoencoder.fit(data, to_categorical(data), epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    autoencoder.save(model_file)
    hist = pd.DataFrame(hist.history)
    with open(model_file + '.loss', mode='w') as f:
        hist.to_csv(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains protein autoencoder from encoded proteins')
    parser.add_argument('input', help='input file contains seqeunce of proteins generated from encode_proteins.py')
    parser.add_argument('model', help='model file (or trained protein autoencoder)')
    args = parser.parse_args()
    main(args.input, args.model)
