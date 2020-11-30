#!/usr/bin/env python3
"""
module 3-variational
contains function autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Function that creates a variational autoencoder. The autoencoder model
    is compiled using adam optimization and binary cross-entropy loss.
    All layers use a relu activation except for the mean and log variance
    layers in the encoder, which uses None, and the last layer in the decoder,
    which uses sigmoid.

    Arguments:
        input_dims: `int`, dimensions of the model input.
        hidden_layers: `list`, with the number of nodes
            for each hidden layer in the encoder, respectively.
            the hidden layers should be reversed for the decoder.
        latent_dims: `int`, dimensions of the latent space representation.

    Returns:
        encoder, decoder, auto: `tuple`,
            encoder is the encoder model, which outputs the latent
                representation, the mean, and the log variance,
                respectively.
            decoder is the decoder model.
            auto is the full autoencoder model.
    """
    # This is our input placeholder
    inputs = keras.Input(shape=(input_dims, ))

    encoder_layers = keras.layers.Dense(units=hidden_layers[0],
                                        activation='relu')(inputs)

    for layer in hidden_layers[1:]:
        encoder_layers = keras.layers.Dense(units=layer,
                                            activation='relu')(encoder_layers)

    z_mean = keras.layers.Dense(latent_dims)(encoder_layers)
    z_log_sigma = keras.layers.Dense(latent_dims)(encoder_layers)

    def sampling(inputs):
        """
        Reparametrization trick: Sample from a normal distribution ∼N(0, 1)
        """
        z_mean, z_log_sigma = inputs
        z_mean_shape = keras.backend.shape(z_mean)
        epsilon = keras.backend.random_normal(shape=z_mean_shape)
        z = z_mean + keras.backend.exp(0.5 * z_log_sigma) * epsilon
        return z

    z = keras.layers.Lambda(sampling,
                            output_shape=(latent_dims, ))([z_mean,
                                                           z_log_sigma])

    # This is our encoded input placeholder
    encoded_inputs = keras.Input(shape=(latent_dims, ))
    decoder_layers = keras.layers.Dense(units=hidden_layers[-1],
                                        activation='relu')(encoded_inputs)

    for layer in reversed(hidden_layers[:-1]):
        decoder_layers = keras.layers.Dense(units=layer,
                                            activation='relu')(decoder_layers)

    # "decoded" is the lossy reconstruction of the input
    decoded = keras.layers.Dense(units=input_dims,
                                 activation='sigmoid')(decoder_layers)

    # Next model maps an input to its encoded representation
    # also called latent space representation or code
    encoder = keras.models.Model(inputs=inputs,
                                 outputs=[z, z_mean, z_log_sigma],
                                 name="encoder")
    # Next model maps a encoded representation
    # to its lossy reconstruction of the input
    decoder = keras.models.Model(inputs=encoded_inputs, outputs=decoded,
                                 name="decoder")

    code = encoder(inputs)[0]
    outputs = decoder(code)

    # This model generates new instarnces that look like that they were
    # sampled from the training set
    auto = keras.models.Model(inputs=inputs, outputs=outputs,
                              name="autoencoder")

    def variational_autoencoder_loss(inputs, outputs):
        """variational autoencoder loss function implementation"""
        reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
        reconstruction_loss *= input_dims
        kl_loss = 1 + z_log_sigma - keras.backend.square(z_mean)\
            - keras.backend.exp(z_log_sigma)
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
        return vae_loss

    auto.compile(optimizer='Adam', loss=variational_autoencoder_loss)

    return encoder, decoder, auto
