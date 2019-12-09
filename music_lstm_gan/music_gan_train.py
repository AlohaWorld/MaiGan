# coding=utf-8

import os
from datetime import datetime

import tensorflow as tf
import tensorflow_probability as tfp
from magenta import music as mm

import configs
import data
from gan_loss import wasserstein_d_update, wasserstein_g_update
from utils import initialize_uninitialized, load_music_vae

flags = tf.app.flags
logging = tf.logging
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'examples_path', None,
    'Path to a TFRecord file of NoteSequence examples. Overrides the config.')
flags.DEFINE_bool(
    'cache_dataset', True,
    'Whether to cache the dataset in memory for improved training speed. May '
    'cause memory errors for very large datasets.')
flags.DEFINE_integer(
    'num_data_threads', 4,
    'The number of data preprocessing threads.')
flags.DEFINE_string(
    'run_dir', None,
    'Path where checkpoints and summary events will be located during '
    'training and evaluation.')
flags.DEFINE_string(
    'config', '',
    'The name of the config to use.')
flags.DEFINE_string(
    'load_model', None,
    'folder of saved model that you wish to continue training, default: None')
flags.DEFINE_string(
    'optimizer', "RMSProp",
    'specify optimizer, default: RMSProp')
flags.DEFINE_float(
    'clipping_value', 0.01,
    'specify clipping value, default: 0.01')
flags.DEFINE_string(
    'output_dir', 'output/generate/',
    'The directory where MIDI files will be saved to.')
flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged: '
    'DEBUG, INFO, WARN, ERROR, or FATAL.')
flags.DEFINE_bool(
    'is_transform', False,
    'whether or not transform from MusicVAE')

# TODO 硬编码
# lots of configurations about MusicVAE for transforming learning
# music_vae_file = '/home/ting/project/dl4m/dataset/pre-trained/hierdec-mel_16bar.tar'
music_vae_file = '/home/gaoting/dl4m/music_vae/pre-trained/hierdec-mel_16bar.tar'
config_music_vae = configs.CONFIG_MAP['vae_hierdec-mel_16bar']

# GPU's configuration
gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True


# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
# gpu_config = tf.ConfigProto(gpu_options=gpu_options)


# Should not be called from within the graph to avoid redundant summaries.
def _trial_summary(hparams, examples_path, output_dir):
    """Writes a tensorboard text summary of the trial."""

    examples_path_summary = tf.summary.text(
        'examples_path', tf.constant(examples_path, name='examples_path'),
        collections=[])

    hparams_dict = hparams.values()

    # Create a markdown table from hparams.
    header = '| Key | Value |\n| :--- | :--- |\n'
    keys = sorted(hparams_dict.keys())
    lines = ['| %s | %s |' % (key, str(hparams_dict[key])) for key in keys]
    hparams_table = header + '\n'.join(lines) + '\n'

    hparam_summary = tf.summary.text(
        'hparams', tf.constant(hparams_table, name='hparams'), collections=[])

    with tf.Session(config=gpu_config) as sess:
        writer = tf.summary.FileWriter(output_dir, graph=sess.graph)
        writer.add_summary(examples_path_summary.eval())
        writer.add_summary(hparam_summary.eval())
        writer.close()


def _get_input_tensors(dataset, config):
    batch_size = config.hparams.batch_size
    iterator = dataset.make_one_shot_iterator()
    input_sequence, output_sequence, _, sequence_length = iterator.get_next()
    input_sequence.set_shape(
        [batch_size, None, config.data_converter.input_depth])
    sequence_length.set_shape([batch_size] + sequence_length.shape[1:].as_list())
    output_sequence.set_shape([batch_size, None, config.data_converter.output_depth])

    return input_sequence, output_sequence, sequence_length


def train(train_dir,
          config,
          dataset_fn,
          load_model=None):
    """Train

    :param train_dir: folder saved during training
    :param config: MusicGAN configuration
    :param dataset_fn: read dataset related function
    :param load_model: folder of saved model that you wish to continue training
    """
    step = 0

    if load_model is not None:
        checkpoints_dir = train_dir + "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
        checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
        meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
        step = int(meta_graph_path.split("-")[2].split(".")[0])
    else:
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoints_dir = train_dir + "checkpoints/{}".format(current_time)
        tf.gfile.MakeDirs(train_dir)

    if step == 0:
        # save configuration of parameters
        _trial_summary(config.hparams, config.train_examples_path, checkpoints_dir)

    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()

        # build MusicGAN model
        music_gan = config.model

        if FLAGS.is_transform:
            _, _, _, generator = load_music_vae(music_vae_file, config_music_vae, sess)
            music_gan.generator = generator

        music_gan.build(
            config.hparams,
            config.data_converter.output_depth,
            is_training=True,
            is_transform=FLAGS.is_transform)

        input_sequence, output_sequence, sequence_length = _get_input_tensors(dataset_fn(), config)
        input_sequence = tf.to_float(input_sequence)
        output_sequence = tf.to_float(output_sequence)
        max_seq_len = config.hparams.max_seq_len
        # for generator's pre train
        x_targets = output_sequence[:, :max_seq_len]
        x_inputs = tf.pad(output_sequence[:, :max_seq_len - 1], [(0, 0), (1, 0), (0, 0)])
        x_length = tf.minimum(sequence_length, max_seq_len)

        # loss of pre train
        if config.hparams.is_gen_pre_train:
            g_pre_train_loss, g_pre_train_accuracy = music_gan.gen_pre_train(x_inputs, x_targets, x_length)
        if config.hparams.is_dis_pre_train:
            d_pre_train_loss, d_pre_train_accuracy = music_gan.dis_pre_train(input_sequence, sequence_length)
        # loss
        g_loss, d_loss = music_gan.train(input_sequence, sequence_length)

        # learning rate
        lr = ((config.hparams.learning_rate - config.hparams.min_learning_rate) *
              tf.pow(config.hparams.decay_rate, tf.to_float(music_gan.global_step)) +
              config.hparams.min_learning_rate)
        tf.summary.scalar('learning_rate', lr)

        # optimizer of pre-train
        if config.hparams.is_gen_pre_train:
            g_pre_train_optimizer = tf.train.AdamOptimizer(lr)
            g_pre_train_update = g_pre_train_optimizer.minimize(g_pre_train_loss,
                                                                var_list=music_gan.generator.get_all_variables())
        if config.hparams.is_dis_pre_train:
            d_pre_train_optimizer = tf.train.AdamOptimizer(lr)
            d_pre_train_update = d_pre_train_optimizer.minimize(d_pre_train_loss,
                                                                var_list=music_gan.discriminator.get_all_variables())
        # optimizer
        if FLAGS.optimizer == "RMSProp":
            g_optimizer = tf.train.RMSPropOptimizer(lr)
            d_optimizer = tf.train.RMSPropOptimizer(lr)
        elif FLAGS.optimizer == "Adam":
            g_optimizer = tf.train.AdamOptimizer(lr)
            d_optimizer = tf.train.AdamOptimizer(lr)
        else:
            raise ValueError('Unsupported optimizer: %s' % FLAGS.optimizer)

        # update discriminator's parameter
        d_update = wasserstein_d_update(
            d_loss,
            d_optimizer,
            FLAGS.clipping_value,
            var_list=music_gan.dis_vars)

        # update generator's parameter
        g_update = wasserstein_g_update(
            g_loss,
            g_optimizer,
            global_step=music_gan.global_step,
            var_list=music_gan.gen_vars)

        # merge summary
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(checkpoints_dir, graph)

        saver = tf.train.Saver()

        if step != 0:
            # restore model for continue train
            save_path = tf.train.latest_checkpoint(checkpoints_dir)
            logging.info('Load model from %s...' % save_path)
            saver.restore(sess, save_path)
        else:
            initialize_uninitialized(sess)
            # clip weights of the discriminator to [-clipping_value, clipping_value]
            # clip_dis_var_op = \
            #     [var.assign(
            #         tf.clip_by_value(var, -FLAGS.clipping_value, FLAGS.clipping_value)
            #     ) for var in music_gan.dis_vars]
            # sess.run(clip_dis_var_op)

            # save the most primitive model
            save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=-1)
            logging.info("Model saved in file: %s" % save_path)

            step = 0

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        sess.run(tf.local_variables_initializer())

        # pre train generator
        if config.hparams.is_gen_pre_train:
            for i in range(config.hparams.gen_pre_train_num):
                _, pre_train_accuracy = sess.run([g_pre_train_update, g_pre_train_accuracy])
                logging.info('----------Step %d:------------' % i)
                logging.info(' g_pre_train_accuracy : {}'.format(pre_train_accuracy))
        if config.hparams.is_dis_pre_train:
            # pre train discriminator
            for i in range(config.hparams.dis_pre_train_num):
                _, pre_train_accuracy = sess.run([d_pre_train_update, d_pre_train_accuracy])
                logging.info('----------Step %d:------------' % i)
                logging.info(' d_pre_train_accuracy : {}'.format(pre_train_accuracy))

        try:
            while not coord.should_stop():
                # train discriminator
                for _ in range(config.hparams.dis_repeat_num):
                    sess.run(d_update)
                # train generator
                for _ in range(config.hparams.gen_repeat_num):
                    sess.run(g_update)

                g_loss_value, d_loss_value, summary = (sess.run([g_loss, d_loss, summary_op]))
                train_writer.add_summary(summary, step)
                train_writer.flush()

                # dis_rd_output, dis_fd_output = \
                #     sess.run([music_gan.dis_real_data_output, music_gan.dis_fake_data_output])
                # real_data = sess.run(music_gan.real_data)
                # music_gan.fake_data is logits
                # fake_data = sess.run(tf.nn.softmax(music_gan.fake_data))

                if step % 2 == 0:
                    logging.info('---------Step %d:-----------' % step)
                    logging.info('   g_loss   : {}'.format(g_loss_value))
                    logging.info('   d_loss   : {}'.format(d_loss_value))
                # save model
                if step % 100 == 0:
                    save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                    logging.info("Model saved in file: %s" % save_path)
                # generate music
                if step % 50 == 0:
                    basename = os.path.join(
                        FLAGS.output_dir,
                        '%s-*.mid' % FLAGS.config, )
                    fake_data = music_gan.fake_data

                    sampler = tfp.distributions.OneHotCategorical(
                        logits=fake_data, dtype=tf.float32)
                    fake_data = sampler.sample()

                    music = config.data_converter.to_items(sess.run(fake_data)[:1])[0]
                    mm.sequence_proto_to_midi_file(music, basename.replace('*', '%04d' % step))

                step += 1

        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
            logging.info("Model saved in file: %s" % save_path)
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)


def run(config_map, tf_file_reader=tf.data.TFRecordDataset):
    if not FLAGS.run_dir:
        raise ValueError('Invalid run directory: %s' % FLAGS.run_dir)
    train_dir = os.path.join(os.path.expanduser(FLAGS.run_dir), 'train/')
    if FLAGS.config is None or FLAGS.config not in config_map:
        raise ValueError('Invalid config: %s' % FLAGS.config)

    config = config_map[FLAGS.config]
    # get data file from example
    config_update_map = {}
    if FLAGS.examples_path:
        config_update_map['%s_examples_path' % 'train'] = os.path.expanduser(FLAGS.examples_path)
    config = configs.update_config(config, config_update_map)

    def dataset_fn():
        """Get input tensors from dataset for training or evaluation."""
        return data.get_dataset(
            config,
            tf_file_reader=tf_file_reader,
            num_threads=FLAGS.num_data_threads,
            is_training=True,
            cache_dataset=FLAGS.cache_dataset)

    train(
        train_dir,
        config=config,
        dataset_fn=dataset_fn,
        load_model=FLAGS.load_model)


def main(unused_argv):
    tf.logging.set_verbosity(FLAGS.log)
    run(configs.CONFIG_MAP)


def console_entry_point():
    tf.app.run(main)


if __name__ == '__main__':
    console_entry_point()
