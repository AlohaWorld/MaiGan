# coding=utf-8
"""Configurations for music_lstm_gan models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf
from magenta.common import merge_hparams
from magenta.models.music_vae import MusicVAE
from tensorflow.contrib.training import HParams

import base_models
import data
import discriminator
import generator
import lstm_models
from base_models import MusicGan


class Config(collections.namedtuple(
    'Config',
    ['model', 'hparams', 'note_sequence_augmenter', 'data_converter',
     'train_examples_path', 'eval_examples_path'])):

    def values(self):
        return self._asdict()


Config.__new__.__defaults__ = (None,) * len(Config._fields)


def update_config(config, update_dict):
    config_dict = config.values()
    config_dict.update(update_dict)
    return Config(**config_dict)


CONFIG_MAP = {}

# 16-bar Melody Models
mel_16bar_converter = data.OneHotMelodyConverter(
    skip_polyphony=False,
    max_bars=100,  # Truncate long melodies before slicing.
    slice_bars=16,
    steps_per_quarter=4)

CONFIG_MAP['gl_dl_16bar'] = Config(
    model=MusicGan(
        generator.LstmGenerator(),
        discriminator.LstmDiscriminator()),
    hparams=merge_hparams(
        base_models.get_default_hparams(),
        HParams(
            batch_size=64,
            max_seq_len=256,
            z_size=512,

            # GAN
            loss_fn='wasserstein',

            # generator
            gen_rnn_size=[2048],

            # discriminator
            dis_rnn_size=[1024, 1024],
            dis_repeat_num=5,
        )),
    note_sequence_augmenter=None,
    data_converter=mel_16bar_converter,
    train_examples_path=None,
    eval_examples_path=None,
)

CONFIG_MAP['gl_db-l_16bar'] = Config(
    model=MusicGan(
        generator.LstmGenerator(),
        discriminator.BidirectionalLstmDiscriminator()),
    hparams=merge_hparams(
        base_models.get_default_hparams(),
        HParams(
            batch_size=64,
            max_seq_len=256,
            z_size=512,

            # GAN
            loss_fn='wasserstein',

            # generator
            gen_rnn_size=[1024, 1024],

            # discriminator
            dis_rnn_size=[512, 512],
            dis_repeat_num=5,
        )),
    note_sequence_augmenter=None,
    data_converter=mel_16bar_converter,
    train_examples_path=None,
    eval_examples_path=None,
)

CONFIG_MAP['gl_dc_16bar'] = Config(
    model=MusicGan(
        generator.LstmGenerator(),
        discriminator.ThreeDimensionalCNNDiscriminator()),
    hparams=merge_hparams(
        base_models.get_default_hparams(),
        HParams(
            batch_size=64,
            max_seq_len=256,
            z_size=512,

            # GAN
            loss_fn='wasserstein',

            # generator
            gen_rnn_size=[1024, 1024],

            # discriminator
            dis_norm='batch_norm',
            dis_activation=tf.nn.leaky_relu,
            dis_repeat_num=5,
        )),
    note_sequence_augmenter=None,
    data_converter=mel_16bar_converter,
    train_examples_path=None,
    eval_examples_path=None,
)

CONFIG_MAP['gl_d2c_16bar'] = Config(
    model=MusicGan(
        generator.LstmGenerator(),
        discriminator.TwoDimensionalCNNDiscriminator()),
    hparams=merge_hparams(
        base_models.get_default_hparams(),
        HParams(
            batch_size=64,
            max_seq_len=256,
            z_size=512,

            # GAN
            loss_fn='wasserstein',

            # generator
            gen_rnn_size=[1024, 1024],

            # discriminator
            dis_norm='batch_norm',
            dis_activation=tf.nn.relu,
            dis_dropout_keep_prob=0.75,
            dis_repeat_num=5,
            dis_filter_sizes=[1, 2, 4, 8, 16, 24, 32, 40, 48],
            dis_num_filters=[100, 200, 200, 200, 200, 50, 100, 50, 100],
        )),
    note_sequence_augmenter=None,
    data_converter=mel_16bar_converter,
    train_examples_path=None,
    eval_examples_path=None,
)

# ========================== for MusicVAE ==========================

CONFIG_MAP['vae_hierdec-mel_16bar'] = Config(
    model=MusicVAE(
        lstm_models.BidirectionalLstmEncoder(),
        generator.HierarchicalLstmGeneratorFromMusicVAE(
            generator.CategoricalLstmDecoderFromMusicVAE(),
            level_lengths=[16, 16],
            disable_autoregression=True)),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=64,
            max_seq_len=256,
            z_size=512,
            enc_rnn_size=[2048, 2048],
            dec_rnn_size=[1024, 1024],
            free_bits=256,
            max_beta=0.2,
        )),
    note_sequence_augmenter=None,
    data_converter=mel_16bar_converter,
    train_examples_path=None,
    eval_examples_path=None,
)

# ============================== END ==============================

# =========================== pre-train ===========================

CONFIG_MAP['glp_dl_16bar'] = Config(
    model=MusicGan(
        generator.LstmGeneratorPreTrainable(),
        discriminator.LstmDiscriminator()),
    hparams=merge_hparams(
        base_models.get_default_hparams(),
        HParams(
            batch_size=64,
            max_seq_len=256,
            z_size=512,

            # GAN
            loss_fn='wasserstein',
            is_gen_pre_train=True,
            is_dis_pre_train=True,

            # generator
            gen_rnn_size=[2048],

            # discriminator
            dis_rnn_size=[1024, 1024],
            dis_repeat_num=5,
        )),
    note_sequence_augmenter=None,
    data_converter=mel_16bar_converter,
    train_examples_path=None,
    eval_examples_path=None,
)

CONFIG_MAP['glp_db-l_16bar'] = Config(
    model=MusicGan(
        generator.LstmGeneratorPreTrainable(),
        discriminator.BidirectionalLstmDiscriminator()),
    hparams=merge_hparams(
        base_models.get_default_hparams(),
        HParams(
            batch_size=64,
            max_seq_len=256,
            z_size=512,

            # GAN
            loss_fn='wasserstein',
            is_gen_pre_train=True,
            is_dis_pre_train=True,

            # generator
            gen_rnn_size=[1024, 1024],

            # discriminator
            dis_rnn_size=[512, 512],
            dis_repeat_num=5,
        )),
    note_sequence_augmenter=None,
    data_converter=mel_16bar_converter,
    train_examples_path=None,
    eval_examples_path=None,
)

CONFIG_MAP['glp_dc_16bar'] = Config(
    model=MusicGan(
        generator.LstmGeneratorPreTrainable(),
        discriminator.ThreeDimensionalCNNDiscriminator()),
    hparams=merge_hparams(
        base_models.get_default_hparams(),
        HParams(
            batch_size=64,
            max_seq_len=256,
            z_size=512,

            # GAN
            loss_fn='wasserstein',
            is_gen_pre_train=True,
            is_dis_pre_train=True,

            # generator
            gen_rnn_size=[1024, 1024],
            gen_pre_train_num=50,

            # discriminator
            dis_norm='batch_norm',
            dis_activation=tf.nn.leaky_relu,
            dis_pre_train_num=500,
            dis_repeat_num=2,
        )),
    note_sequence_augmenter=None,
    data_converter=mel_16bar_converter,
    train_examples_path=None,
    eval_examples_path=None,
)

CONFIG_MAP['glp_d2c_16bar'] = Config(
    model=MusicGan(
        generator.LstmGeneratorPreTrainable(),
        discriminator.TwoDimensionalCNNDiscriminator()),
    hparams=merge_hparams(
        base_models.get_default_hparams(),
        HParams(
            batch_size=64,
            max_seq_len=256,
            z_size=512,

            # GAN
            loss_fn='wasserstein',
            is_gen_pre_train=True,
            is_dis_pre_train=True,

            # generator
            gen_rnn_size=[1024, 1024],
            gen_repeat_num=10,

            # discriminator
            # dis_norm='batch_norm',
            dis_norm=None,
            dis_activation=tf.nn.relu,
            dis_dropout_keep_prob=0.75,
            dis_pre_train_num=50,
            dis_repeat_num=1,
            dis_filter_sizes=[1, 4, 8, 16, 24, 32, 40, 48],
            dis_num_filters=[50, 100, 100, 100, 25, 50, 25, 50],
        )),
    note_sequence_augmenter=None,
    data_converter=mel_16bar_converter,
    train_examples_path=None,
    eval_examples_path=None,
)

CONFIG_MAP['ghlp_d2c_16bar'] = Config(
    model=MusicGan(
        generator.HierarchicalLstmGeneratorFromMusicVAE(
            generator.CategoricalLstmDecoderFromMusicVAE(),
            level_lengths=[16, 16],
            disable_autoregression=True),
        discriminator.TwoDimensionalCNNDiscriminator()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=64,
            max_seq_len=256,
            z_size=512,

            bar_num=16,  # specify the number of bar per a music
            beat_num=16,  # specify the number of beat per a bar
            tracks_num=1,  # specify the number of audio track

            # GAN
            loss_fn='wasserstein',
            is_gen_pre_train=True,
            is_dis_pre_train=True,

            # generator
            dec_rnn_size=[1024, 1024],
            gen_pre_train_num=150,
            gen_repeat_num=30,

            # discriminator
            dis_norm='batch_norm',
            dis_activation=tf.nn.relu,
            dis_dropout_keep_prob=0.75,
            dis_pre_train_num=50,
            dis_repeat_num=1,

            dis_filter_sizes=[1, 4, 8, 16, 24, 32, 40, 48],
            dis_num_filters=[50, 100, 100, 100, 25, 50, 25, 50],
        )),
    note_sequence_augmenter=None,
    data_converter=mel_16bar_converter,
    train_examples_path=None,
    eval_examples_path=None,
)

# ============================== END ==============================
