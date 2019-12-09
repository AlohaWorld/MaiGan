# coding=utf-8
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from magenta import music as mm

import configs
from utils import sample_and_save_z, load_z_from_file

flags = tf.app.flags
logging = tf.logging
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'run_dir', None,
    'Path where checkpoints and summary events will be located during '
    'training and evaluation. Separate subdirectories `train` and `eval` '
    'will be created within this directory.')
flags.DEFINE_string(
    'config', '',
    'The name of the config to use.')
flags.DEFINE_string(
    'load_model', None,
    'folder of saved model that you wish to use, default: None')
flags.DEFINE_string(
    'mode', 'sample',
    'Generate mode (either `sample` or `interpolate`).')
flags.DEFINE_integer(
    'num_outputs', 10,
    'Specify the number of generated music.')
flags.DEFINE_string(
    'output_dir', 'output/generate/',
    'The directory where MIDI files will be saved to.')
flags.DEFINE_string(
    'z_file', None,
    'Specify a file that saves lots of latent variables(z). '
    'Note that the number of z must be equal to the number of generated music.')
flags.DEFINE_string(
    'z_output', 'output/z_file/',
    'Specify a folder that saves lots of latent variables(z). ')
flags.DEFINE_bool(
    'is_transform', False,
    'whether or not transform from MusicVAE')

flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')

graph = tf.get_default_graph()

# TODO 硬编码
config_music_vae = configs.CONFIG_MAP['vae_hierdec-mel_16bar']


def load_music_gan(music_gan_file, config_music_gan):
    """load pre-trained music_gan

    :param music_gan_file: a checkpoint_path about MusicGAN
    :param config_music_gan: a Configuration file about MusicGAN
    :return: MusicGAN
    """
    checkpoint_path = os.path.expanduser(music_gan_file)
    if tf.gfile.IsDirectory(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    else:
        checkpoint_path = checkpoint_path
    config = config_music_gan
    config.data_converter.set_mode('infer')

    with graph.as_default():
        # build MusicGAN model
        music_gan = config.model
        music_gan.build(
            config.hparams,
            config.data_converter.output_depth,
            is_training=False)

        if FLAGS.is_transform:
            music_vae = config_music_vae.model
            music_vae.build(
                config_music_vae.hparams,
                config_music_vae.data_converter.output_depth,
                is_training=False)
            music_gan.generator = music_vae.decoder

        batch_size = config.hparams.batch_size
        z_size = config.hparams.z_size

        max_length = tf.placeholder(tf.int32, shape=())
        z_inputs = tf.placeholder(tf.float32, shape=[batch_size, z_size])

        generator, _ = music_gan.generator.generate(z_inputs, max_length)

        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)

    return generator, z_inputs, max_length, sess


def sample(generator, z_inputs, max_length):
    """Use generator to generate music.

    Note that z and result are tensor objects.

    :param generator: specify generator
    :param z_inputs: Batch of latent vectors, used to guide music generation
        [n, z_size]
    :param max_length: specify music max length
    :return: Generate music samples.
        [n, max_length, output_depth]
    """
    return generator.generate(z_inputs, max_length)


def run(config_map):
    run_dir = FLAGS.run_dir
    if run_dir is None:
        raise ValueError('You must specify run_dir')
    load_model = FLAGS.load_model
    if load_model is None:
        raise ValueError('You must specify load_model')
    if FLAGS.config is None or FLAGS.config not in config_map:
        raise ValueError('Invalid config: %s' % FLAGS.config)
    config = config_map[FLAGS.config]

    # load MusicGAN
    train_dir = os.path.join(os.path.expanduser(run_dir), 'train/')
    save_path = train_dir + "checkpoints/" + load_model.lstrip("checkpoints/")
    generator, z_inputs, max_length, sess = load_music_gan(save_path, config)

    mode = FLAGS.mode
    if mode == 'sample':
        z_size = config.hparams.z_size
        num_outputs = FLAGS.num_outputs
        z_file = FLAGS.z_file
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        if z_file is None:
            output_file = os.path.join(FLAGS.z_output, current_time)
            logging.info('Save latent variables(z) to file:%s!' % output_file)
            z = sample_and_save_z(z_size, num_outputs, output_file)
        else:
            z = load_z_from_file(z_file)
            num_outputs = z.shape[0]

        batch_size = config.hparams.batch_size
        batch_pad_amt = -num_outputs % batch_size
        z = np.pad(z, [(0, batch_pad_amt), (0, 0)], mode='constant')

        outputs = sess.run(
            generator,
            feed_dict={
                max_length: config.hparams.max_seq_len,
                z_inputs: z,
            })

        with graph.as_default():
            sampler = tfp.distributions.OneHotCategorical(
                logits=outputs, dtype=tf.float32)
            outputs = sampler.sample()

            samples = outputs[:num_outputs]
            results = config.data_converter.to_items(sess.run(samples))

        basename = os.path.join(
            FLAGS.output_dir,
            '%s_%s-*-of-%03d.mid' %
            (FLAGS.config, current_time, num_outputs))

        logging.info('Outputting %d files as `%s`...', FLAGS.num_outputs, basename)

        for i, ns in enumerate(results):
            mm.sequence_proto_to_midi_file(ns, basename.replace('*', '%03d' % i))

        logging.info('Done.')
    elif mode == 'interpolate':
        # TODO interpolate
        pass
    else:
        raise ValueError('Invalid mode: %s!' % mode)


def main(unused_argv):
    tf.logging.set_verbosity(FLAGS.log)
    run(configs.CONFIG_MAP)


def console_entry_point():
    tf.app.run(main)


if __name__ == '__main__':
    console_entry_point()
