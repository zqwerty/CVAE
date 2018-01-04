import tensorflow as tf


def gaussian_kld(mu_1, logvar_1, mu_2, logvar_2):
    kld = -0.5 * tf.reduce_sum(1 + (logvar_1 - logvar_2)
                               - tf.div(tf.pow(mu_2 - mu_1, 2), tf.exp(logvar_2))
                               - tf.div(tf.exp(logvar_1), tf.exp(logvar_2)), reduction_indices=1)
    return kld


def sample_gaussian(mu, logvar):
    epsilon = tf.random_normal(tf.shape(logvar), name="epsilon")
    std = tf.exp(0.5 * logvar)
    z = mu + tf.multiply(std, epsilon)
    return z
