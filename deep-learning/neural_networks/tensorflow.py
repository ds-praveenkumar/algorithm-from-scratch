import tensorflow as tf
import numpy as np



def get_ae( n_in: int , n_out: int, dropout: float):
    """
    Builds an Autoencoder 

    """
    x_in = tf.keras.Input( shape=n_in.shape[1])

    # encoder
    x = tf.keras.layers.Dense(512, activation='relu')(x_in)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense( 64, activation='relu')(x)

    # decoder
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense( 256, activation='relu')(x)
    out = tf.keras.layers.Dense(512, activation='relu')(x)

    model = tf.keras.Model( x_in, out)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

get_ae(7,7,0.2)