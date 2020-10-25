import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import tensorflow as tf
from keras import backend as K
import argparse
import os


def main(input_file, output_file, model_file):
    assert os.path.exists(input_file)
    assert os.path.exists(model_file)

    print("Loading data")
    data = pd.read_csv(input_file, header=None)
    data = data.values
    sdata = data.shape
    data = np.reshape(data, [sdata[0], sdata[1], 1])
    print("Loaded data")
    print(sdata)

    print("Loading model")
    autoencoder = tf.keras.models.load_model(model_file)
    autoencoder.summary()

    input_layer = autoencoder.input
    latent_layer = autoencoder.get_layer(name="latent_vector")
    latent_output = latent_layer.output
    eval_lat = K.function([input_layer, K.learning_phase()], latent_output)

    batch = 2048
    out = eval_lat([data[0:batch, :], 1.])
    print(out.shape)

    for i in range(batch, data.shape[0], batch):
        lval = eval_lat([data[i:i + batch, :], 1.])
        out = np.concatenate((out, lval))
        print(out.shape)

    np.savetxt(output_file, out, delimiter=",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate latent space for encoded proteins')
    parser.add_argument('input', help='input file obtained from encode_proteins.py')
    parser.add_argument('output', help='output file contains latent space for proteins')
    parser.add_argument('model', help='model file')
    args = parser.parse_args()
    main(args.input, args.output, args.model)
