# coding=utf-8

import tensorflow as tf

import lstm_utils
from ops import get_normalization, dense, conv3d, conv2d
from utils import highway

rnn = tf.contrib.rnn


class BaseDiscriminator(object):
    """Abstract Discriminator"""

    def build(self, hparams, is_training=True, name_or_scope='discriminator'):
        """Builder method for BaseDiscriminator."""
        pass

    def discriminate(self, data_inputs, sequence_length):
        """Discriminate input data.

        :param data_inputs: dataset that needs to discriminate
            [batch_size, time_step, pitch]
        :param sequence_length
            [batch_size, 1]
        :return: discriminator result
            [batch_size, 1]
        """
        pass

    def get_all_variables(self):
        """Get all trainable variables in discriminator"""
        return tf.global_variables(scope=self._name_or_scope)

    def pre_train(self, data_inputs, data_labels, sequence_length):
        """Pre-train.

        :param data_inputs:
            [batch_size*2, time_step, pitch]
        :param data_labels:
            [batch_size*2, 2]
        :param sequence_length:
            [batch_size, 1]
        :return: two operations about pre-train
            (pre_train_loss, pre_train_accuracy)
        """
        pass


class LstmDiscriminator(BaseDiscriminator):
    """LSTM discriminator."""

    def build(self, hparams, is_training=True, name_or_scope='discriminator'):
        self._name_or_scope = name_or_scope

        tf.logging.info('\nDiscriminator Cells (unidirectional):\n'
                        '  units: %s\n',
                        hparams.dis_rnn_size)

        self._cell = lstm_utils.rnn_cell(
            hparams.dis_rnn_size, hparams.dropout_keep_prob,
            hparams.dis_residual, is_training)

    def discriminate(self, data_inputs, sequence_length):
        # Convert to time-major.
        sequence = tf.transpose(data_inputs, [1, 0, 2])

        with tf.variable_scope(self._name_or_scope, reuse=tf.AUTO_REUSE):
            outputs, _ = tf.nn.dynamic_rnn(
                self._cell,
                sequence,
                sequence_length,
                dtype=tf.float32,
                time_major=True,
                scope=self._name_or_scope)
            mu = tf.layers.dense(
                outputs[-1],
                1,
                kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                name='mu_dense')

        return mu

    def pre_train(self, data_inputs, data_labels, sequence_length):
        # Convert to time-major.
        sequence = tf.transpose(data_inputs, [1, 0, 2])

        with tf.variable_scope(self._name_or_scope, reuse=tf.AUTO_REUSE):
            outputs, _ = tf.nn.dynamic_rnn(
                self._cell,
                sequence,
                sequence_length,
                dtype=tf.float32,
                time_major=True,
                scope=self._name_or_scope)

            p_logits = tf.layers.dense(outputs[-1], 2, name='logits_dense')
            pre_train_loss = tf.nn.softmax_cross_entropy_with_logits(logits=p_logits, labels=data_labels)

            truth = tf.argmax(data_labels, axis=1)
            predictions = tf.argmax(p_logits, axis=1)
            pre_train_accuracy = tf.metrics.accuracy(truth, predictions)

        return pre_train_loss, pre_train_accuracy


class BidirectionalLstmDiscriminator(BaseDiscriminator):
    """Bi-LSTM discriminator."""

    def build(self, hparams, is_training=True, name_or_scope='discriminator'):
        self._name_or_scope = name_or_scope

        tf.logging.info('\nBi-Discriminator Cells:\n'
                        '  units: %s\n',
                        hparams.dis_rnn_size)

        cells_fw = []
        cells_bw = []
        for i, layer_size in enumerate(hparams.dis_rnn_size):
            cells_fw.append(
                lstm_utils.rnn_cell(
                    [layer_size], hparams.dropout_keep_prob,
                    hparams.dis_residual, is_training))
            cells_bw.append(
                lstm_utils.rnn_cell(
                    [layer_size], hparams.dropout_keep_prob,
                    hparams.dis_residual, is_training))

        self._cells = (cells_fw, cells_bw)

    def discriminate(self, data_inputs, sequence_length):
        cells_fw, cells_bw = self._cells

        with tf.variable_scope(self._name_or_scope, reuse=tf.AUTO_REUSE):
            _, states_fw, states_bw = rnn.stack_bidirectional_dynamic_rnn(
                cells_fw,
                cells_bw,
                data_inputs,
                sequence_length=sequence_length,
                time_major=False,
                dtype=tf.float32,
                scope=self._name_or_scope)
            # Note we access the outputs (h) from the states since the backward
            # ouputs are reversed to the input order in the returned outputs.
            last_h_fw = states_fw[-1][-1].h
            last_h_bw = states_bw[-1][-1].h

            mu = tf.layers.dense(
                tf.concat([last_h_fw, last_h_bw], 1),
                1,
                kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                name='mu_dense')

        return mu

    def pre_train(self, data_inputs, data_labels, sequence_length):
        cells_fw, cells_bw = self._cells

        with tf.variable_scope(self._name_or_scope, reuse=tf.AUTO_REUSE):
            _, states_fw, states_bw = rnn.stack_bidirectional_dynamic_rnn(
                cells_fw,
                cells_bw,
                data_inputs,
                sequence_length=sequence_length,
                time_major=False,
                dtype=tf.float32,
                scope=self._name_or_scope)
            # Note we access the outputs (h) from the states since the backward
            # ouputs are reversed to the input order in the returned outputs.
            last_h_fw = states_fw[-1][-1].h
            last_h_bw = states_bw[-1][-1].h

            p_logits = tf.layers.dense(tf.concat([last_h_fw, last_h_bw], 1), 2, name='logits_dense')
            pre_train_loss = tf.nn.softmax_cross_entropy_with_logits(logits=p_logits, labels=data_labels)

            truth = tf.argmax(data_labels, axis=1)
            predictions = tf.argmax(p_logits, axis=1)
            pre_train_accuracy = tf.metrics.accuracy(truth, predictions)

        return pre_train_loss, pre_train_accuracy


class ThreeDimensionalCNNDiscriminator(BaseDiscriminator):
    """3D-CNN discriminator."""

    def build(self, hparams, is_training=True, name_or_scope='discriminator'):
        self._name_or_scope = name_or_scope

        norm = get_normalization(hparams.dis_norm, is_training)
        activation = hparams.dis_activation
        self._conv_layer = lambda i, f, k, s: activation(norm(conv3d(i, f, k, s)))
        self._tracks_num = hparams.tracks_num
        self._bar_num = hparams.bar_num
        self._beat_num = hparams.beat_num

        self._batch_size = hparams.batch_size

    def discriminate(self, data_inputs, sequence_length=None):
        batch_size = data_inputs.shape.as_list()[0]
        pitch_num = data_inputs.shape.as_list()[2]

        # h shape [batch_size, bars, beats, pitch, tracks_nums]
        h = tf.reshape(data_inputs, shape=[batch_size, self._bar_num, self._beat_num, pitch_num, self._tracks_num])

        with tf.variable_scope(self._name_or_scope, reuse=tf.AUTO_REUSE):
            # Pitch-time private network
            with tf.variable_scope('pitch_time_private'):
                s1 = [self._conv_layer(h, 16, (1, 1, 6), (1, 1, 6))  # 16, 16, 15
                      for _ in range(self._tracks_num)]
                s1 = [self._conv_layer(s1[i], 32, (1, 2, 1), (1, 2, 1))  # 16, 8, 15
                      for i in range(self._tracks_num)]

            # Time-pitch private network
            with tf.variable_scope('time_pitch_private'):
                s2 = [self._conv_layer(h, 16, (1, 2, 1), (1, 2, 1))  # 16, 8, 90
                      for _ in range(self._tracks_num)]
                s2 = [self._conv_layer(s2[i], 32, (1, 1, 6), (1, 1, 6))  # 16, 8, 15
                      for i in range(self._tracks_num)]

            h = [tf.concat((s1[i], s2[i]), -1) for i in range(self._tracks_num)]

            # Merged private network
            with tf.variable_scope('merged_private'):
                h = [self._conv_layer(h[i], 64, (1, 1, 1), (1, 1, 1))  # 16, 8, 15
                     for i in range(self._tracks_num)]

            h = tf.concat(h, -1)

            # Shared network
            with tf.variable_scope('shared'):
                h = self._conv_layer(h, 128, (1, 2, 5), (1, 2, 5))  # 16, 4, 3
                h = self._conv_layer(h, 256, (1, 4, 3), (1, 4, 3))  # 16, 1, 1

            h = tf.reshape(h, shape=[h.shape[0], -1])
            h = dense(h, 1, name='h_dense')  # batch_size, 1
        return h

    def pre_train(self, data_inputs, data_labels, sequence_length=None):
        batch_size = self._batch_size * 2
        pitch_num = data_inputs.shape.as_list()[2]

        # h shape [batch_size, bars, beats, pitch, tracks_nums]
        h = tf.reshape(data_inputs, shape=[batch_size, self._bar_num, self._beat_num, pitch_num, self._tracks_num])

        with tf.variable_scope(self._name_or_scope, reuse=tf.AUTO_REUSE):
            # Pitch-time private network
            with tf.variable_scope('pitch_time_private'):
                s1 = [self._conv_layer(h, 16, (1, 1, 6), (1, 1, 6))  # 16, 16, 15
                      for _ in range(self._tracks_num)]
                s1 = [self._conv_layer(s1[i], 32, (1, 2, 1), (1, 2, 1))  # 16, 8, 15
                      for i in range(self._tracks_num)]

            # Time-pitch private network
            with tf.variable_scope('time_pitch_private'):
                s2 = [self._conv_layer(h, 16, (1, 2, 1), (1, 2, 1))  # 16, 8, 90
                      for _ in range(self._tracks_num)]
                s2 = [self._conv_layer(s2[i], 32, (1, 1, 6), (1, 1, 6))  # 16, 8, 15
                      for i in range(self._tracks_num)]

            h = [tf.concat((s1[i], s2[i]), -1) for i in range(self._tracks_num)]

            # Merged private network
            with tf.variable_scope('merged_private'):
                h = [self._conv_layer(h[i], 64, (1, 1, 1), (1, 1, 1))  # 16, 8, 15
                     for i in range(self._tracks_num)]

            h = tf.concat(h, -1)

            # Shared network
            with tf.variable_scope('shared'):
                h = self._conv_layer(h, 128, (1, 2, 5), (1, 2, 5))  # 16, 4, 3
                h = self._conv_layer(h, 256, (1, 4, 3), (1, 4, 3))  # 16, 1, 1

            h = tf.reshape(h, shape=[h.shape[0], -1])
            p_logits = tf.layers.dense(h, 2, name='logits_dense')
            pre_train_loss = tf.nn.softmax_cross_entropy_with_logits(logits=p_logits, labels=data_labels)

            truth = tf.argmax(data_labels, axis=1)
            predictions = tf.argmax(p_logits, axis=1)
            pre_train_accuracy = tf.metrics.accuracy(truth, predictions)

        return pre_train_loss, pre_train_accuracy


class TwoDimensionalCNNDiscriminator(BaseDiscriminator):
    """2D-CNN discriminator.

    To see: https://github.com/LantaoYu/SeqGAN/blob/master/discriminator.py

    # TODO test ==> add "batch_norm" && add pre-train
    """

    def build(self, hparams, is_training=True, name_or_scope='discriminator'):
        self._name_or_scope = name_or_scope

        self._norm = get_normalization(hparams.dis_norm, is_training)
        self._activation = hparams.dis_activation
        self._dropout_keep_prob = hparams.dis_dropout_keep_prob
        self._filter_sizes = hparams.dis_filter_sizes
        self._num_filters = hparams.dis_num_filters

        self._pitch_num = hparams.pitch_num
        self._tracks_num = hparams.tracks_num
        self._seq_len = hparams.max_seq_len

    def discriminate(self, data_inputs, sequence_length=None):
        """Discriminate input data.

        Note: Currently only supports single tracks_number
        """
        # [batch_size, time_step, pitch] ==> [batch_size, time_step, pitch, tracks_num(1)]
        pitch_num = self._pitch_num
        data_inputs = tf.expand_dims(data_inputs, -1)

        # Create a convolution + maxpool layer for each filter size
        with tf.variable_scope(self._name_or_scope, reuse=tf.AUTO_REUSE):
            pooled_outputs = []
            for filter_size, num_filter in zip(self._filter_sizes, self._num_filters):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    conv = conv2d(data_inputs, num_filter, [filter_size, pitch_num], [1, 1])
                    # add norm
                    h = self._activation(self._norm(conv))
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self._seq_len - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = sum(self._num_filters)
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

            # Add highway
            with tf.name_scope("highway"):
                h_highway = highway(h_pool_flat, h_pool_flat.get_shape()[1], 1, 0)

            # Add dropout
            with tf.name_scope("dropout"):
                h_drop = tf.nn.dropout(h_highway, self._dropout_keep_prob)

            h = dense(h_drop, 1)

        return h

    def pre_train(self, data_inputs, data_labels, sequence_length=None):
        # [batch_size, time_step, pitch] ==> [batch_size, time_step, pitch, tracks_num(1)]
        pitch_num = data_inputs.shape.as_list()[2]
        data_inputs = tf.expand_dims(data_inputs, -1)

        # Create a convolution + maxpool layer for each filter size
        with tf.variable_scope(self._name_or_scope, reuse=tf.AUTO_REUSE):
            pooled_outputs = []
            for filter_size, num_filter in zip(self._filter_sizes, self._num_filters):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, pitch_num, 1, num_filter]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    conv = tf.nn.conv2d(
                        data_inputs,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # add norm
                    h = self._activation(self._norm(conv))
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, self._seq_len - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = sum(self._num_filters)
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

            # Add highway
            with tf.name_scope("highway"):
                h_highway = highway(h_pool_flat, h_pool_flat.get_shape()[1], 1, 0)

            # Add dropout
            with tf.name_scope("dropout"):
                h_drop = tf.nn.dropout(h_highway, self._dropout_keep_prob)

            W = tf.Variable(tf.truncated_normal([num_filters_total, 2], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
            scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")

            truth = tf.argmax(data_labels, axis=1)
            predictions = tf.argmax(scores, axis=1)

            pre_train_loss = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=data_labels)
            pre_train_accuracy = tf.metrics.accuracy(truth, predictions)

        return pre_train_loss, pre_train_accuracy
