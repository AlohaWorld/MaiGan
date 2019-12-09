# coding=utf-8

import os
import tarfile

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from backports import tempfile

from ops import dense

tfd = tfp.distributions


def sample_and_save_z(z_size, num, output_file=None):
    """Sample from a multidimensional standard normal distribution,
        and then save them to a specified file

    :param z_size: multidimensional standard normal distribution
    :param num: specify the number of latent variable(z)
    :param output_file: a file used to save result
    :return: A tensor(z)
    """
    loc = [0] * z_size
    scale_diag = [1] * z_size
    mvn = tfd.MultivariateNormalDiag(
        loc=loc,
        scale_diag=scale_diag)
    with tf.Session():
        z = mvn.sample(num).eval()
    z = np.array(z)
    if output_file is not None:
        np.save(output_file, z)
    return z


def load_z_from_file(z_file):
    """Load z from a specified file

    :param z_file: specify a file that saves lots fo latent variable(z)
    :return: A tensor(z)
    """
    z = np.load(z_file)
    return z


def initialize_uninitialized(sess):
    """Use to initialize uninitialized variables

    :param sess: target session
    """
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def load_music_vae(music_vae_file, config_music_vae, sess):
    """load music_vae

    Note that load the decoder from music vae, but the decoder is seen as a generator in MusicGAN.

    :param music_vae_file: load pre-trained music_vae
    :param config_music_vae: specify a configuration used by MusicVAE
    :param sess: Specify a session
    :return: Generator(a class's instance)
    """
    checkpoint_path = os.path.expanduser(music_vae_file)
    if tf.gfile.IsDirectory(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    else:
        checkpoint_path = checkpoint_path

    config = config_music_vae
    batch_size = config.hparams.batch_size
    z_size = config.hparams.z_size

    music_vae = config.model
    music_vae.build(
        config.hparams,
        config.data_converter.output_depth,
        is_training=True)

    max_length = tf.placeholder(tf.int32, shape=())
    z_inputs = tf.placeholder(tf.float32, shape=[batch_size, z_size])

    generator = music_vae.decoder
    all_samples = generator.generate(z_inputs, max_length)

    saver = tf.train.Saver()
    if (os.path.exists(checkpoint_path) and
            tarfile.is_tarfile(checkpoint_path)):
        tf.logging.info('Unbundling checkpoint.')
        with tempfile.TemporaryDirectory() as temp_dir:
            tar = tarfile.open(checkpoint_path)
            tar.extractall(temp_dir)
            # Assume only a single checkpoint is in the directory.
            for name in tar.getnames():
                if name.endswith('.index'):
                    checkpoint_path = os.path.join(temp_dir, name[0:-6])
                    break
            saver.restore(sess, checkpoint_path)
    else:
        saver.restore(sess, checkpoint_path)

    return all_samples, max_length, z_inputs, generator


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.

    see: https://github.com/LantaoYu/SeqGAN/blob/master/discriminator.py

    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(dense(input_, size, name='highway_lin_%d' % idx))

            t = tf.sigmoid(dense(input_, size, name='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return output
