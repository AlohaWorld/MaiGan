# coding=utf-8

import tensorflow as tf
import tensorflow_probability as tfp
from magenta.common import flatten_maybe_padded_sequences
from tensorflow.python.layers import core as layers_core

import lstm_models
import lstm_utils

seq2seq = tf.contrib.seq2seq


class BaseGenerator(object):
    """Abstract Generator"""

    def build(self, hparams, output_depth, is_training=True, name_or_scope='generator'):
        """Builder method for BaseGenerator."""
        pass

    def generate(self, z_inputs, max_length):
        """Use generator to generate musics, according to hidden variable.

        :param z_inputs: hidden variable
            [batch_size, z_size]
        :param max_length: a scalar number
        :return
            [batch_size, time_step, pitch(output_depth)]
        """
        pass

    def get_all_variables(self):
        """Get all trainable variables in generator"""
        return tf.global_variables(scope=self._name_or_scope)


class LstmGenerator(BaseGenerator):
    """LSTM generator"""

    def build(self, hparams, output_depth, is_training=True, name_or_scope='generator'):
        self._name_or_scope = name_or_scope
        self._output_depth = output_depth
        self._output_layer = layers_core.Dense(output_depth, trainable=is_training, name='output_projection')

        tf.logging.info('\nGenerator Cells:\n'
                        '  units: %s\n',
                        hparams.gen_rnn_size)

        self._gen_cell = lstm_utils.rnn_cell(
            hparams.gen_rnn_size,
            hparams.dropout_keep_prob,
            hparams.gen_residual,
            is_training)

    def _sample(self, rnn_output):
        return rnn_output

    def generate(self, z_inputs, max_length):
        batch_size = z_inputs.shape[0]

        start_inputs = tf.zeros([batch_size, self._output_depth], dtype=tf.float32)
        start_inputs = tf.concat([start_inputs, z_inputs], axis=-1)

        initialize_fn = lambda: (tf.zeros([batch_size], tf.bool), start_inputs)
        sample_fn = lambda time, outputs, state: self._sample(outputs)

        end_fn = (lambda x: False)

        def next_inputs_fn(time, outputs, state, sample_ids):
            del outputs
            finished = end_fn(sample_ids)

            # next_inputs is a sample of one-hot, sample_id == log(logits)
            sampler = tfp.distributions.OneHotCategorical(
                logits=sample_ids, dtype=tf.float32)
            tmp = sampler.sample()

            next_inputs = tf.concat([tmp, z_inputs], axis=-1)
            return finished, next_inputs, state

        sampler = seq2seq.CustomHelper(
            initialize_fn=initialize_fn,
            sample_fn=sample_fn,
            next_inputs_fn=next_inputs_fn,
            sample_ids_shape=[self._output_depth],
            sample_ids_dtype=tf.float32)

        with tf.variable_scope(self._name_or_scope, reuse=tf.AUTO_REUSE):
            initial_state = lstm_utils.initial_cell_state_from_embedding(self._gen_cell, z_inputs)

            decoder = lstm_utils.Seq2SeqLstmDecoder(
                self._gen_cell,
                sampler,
                initial_state=initial_state,
                input_shape=start_inputs.shape[1:],
                output_layer=self._output_layer)

            final_output, final_state, final_lengths = seq2seq.dynamic_decode(
                decoder,
                maximum_iterations=max_length,
                swap_memory=True)
        return final_output.sample_id, final_lengths


class CategoricalLstmDecoderFromMusicVAE(lstm_models.CategoricalLstmDecoder):

    def sample(self, n, max_length=None, z=None, c_input=None, temperature=None,
               start_inputs=None, beam_width=None, end_token=None):
        batch_size = n

        start_inputs = tf.zeros([batch_size, self._output_depth], dtype=tf.float32)
        start_inputs = tf.concat([start_inputs, z], axis=-1)

        initialize_fn = lambda: (tf.zeros([batch_size], tf.bool), start_inputs)
        sample_fn = lambda time, outputs, state: outputs

        end_fn = (lambda x: False)

        def next_inputs_fn(time, outputs, state, sample_ids):
            del outputs
            finished = end_fn(sample_ids)

            # next_inputs is a sample of one-hot, sample_id == log(logits)
            sampler = tfp.distributions.OneHotCategorical(
                logits=sample_ids, dtype=tf.float32)
            tmp = sampler.sample()

            next_inputs = tf.concat([tmp, z], axis=-1)
            return finished, next_inputs, state

        sampler = seq2seq.CustomHelper(
            initialize_fn=initialize_fn,
            sample_fn=sample_fn,
            next_inputs_fn=next_inputs_fn,
            sample_ids_shape=[self._output_depth],
            sample_ids_dtype=tf.float32)

        decode_results = self._decode(
            z, helper=sampler, input_shape=start_inputs.shape[1:],
            max_length=max_length)

        return decode_results.samples, decode_results


class HierarchicalLstmGeneratorFromMusicVAE(lstm_models.HierarchicalLstmDecoder):

    def generate(self, z_inputs, max_length):
        batch_size = z_inputs.shape[0]
        all_samples, decode_results = self.sample(batch_size, max_length=max_length, z=z_inputs)
        final_lengths = tf.reduce_sum(decode_results.final_sequence_lengths, axis=1)
        # fixed length
        return all_samples, final_lengths

    def get_all_variables(self):
        variables = []
        variables.extend(tf.global_variables(scope="decoder"))
        variables.extend(tf.global_variables(scope="core_decoder"))
        return variables

    def pre_train(self, x_inputs, x_targets, x_inputs_length, max_length, z_size):
        batch_size = x_inputs.shape[0]
        z_simulated = tf.zeros([batch_size, z_size], dtype=tf.float32)
        pre_train_loss, merged_metric_map, results = self.reconstruction_loss(x_inputs,
                                                                              x_targets,
                                                                              x_inputs_length,
                                                                              z=z_simulated)
        predictions = results.samples
        truth = tf.argmax(x_targets, axis=2)
        pre_train_accuracy = tf.metrics.accuracy(truth, predictions)
        return pre_train_loss, pre_train_accuracy


class LstmGeneratorPreTrainable(BaseGenerator):
    """LSTM generator

    Generator can be pre-trainable in this version.
    """

    def build(self, hparams, output_depth, is_training=True, name_or_scope='generator'):
        self._name_or_scope = name_or_scope
        self._output_depth = output_depth
        self._output_layer = layers_core.Dense(output_depth, trainable=is_training, name='output_projection')

        tf.logging.info('\nGenerator Cells:\n'
                        '  units: %s\n',
                        hparams.gen_rnn_size)

        self._gen_cell = lstm_utils.rnn_cell(
            hparams.gen_rnn_size,
            hparams.gen_dropout_keep_prob,
            hparams.gen_residual,
            is_training)

    def _sample(self, rnn_output):
        return rnn_output

    def generate(self, z_inputs, max_length):
        batch_size = z_inputs.shape[0]

        start_inputs = tf.zeros([batch_size, self._output_depth], dtype=tf.float32)
        start_inputs = tf.concat([start_inputs, z_inputs], axis=-1)

        initialize_fn = lambda: (tf.zeros([batch_size], tf.bool), start_inputs)
        sample_fn = lambda time, outputs, state: self._sample(outputs)
        end_fn = (lambda x: False)

        def next_inputs_fn(time, outputs, state, sample_ids):
            del outputs
            finished = end_fn(sample_ids)

            # next_inputs is a sample of one-hot, sample_id == log(logits)
            sampler = tfp.distributions.OneHotCategorical(
                logits=sample_ids, dtype=tf.float32)
            tmp = sampler.sample()

            next_inputs = tf.concat([tmp, z_inputs], axis=-1)
            return finished, next_inputs, state

        sampler = seq2seq.CustomHelper(
            initialize_fn=initialize_fn,
            sample_fn=sample_fn,
            next_inputs_fn=next_inputs_fn,
            sample_ids_shape=[self._output_depth],
            sample_ids_dtype=tf.float32)

        with tf.variable_scope(self._name_or_scope, reuse=tf.AUTO_REUSE):
            initial_state = lstm_utils.initial_cell_state_from_embedding(self._gen_cell, z_inputs)

            decoder = lstm_utils.Seq2SeqLstmDecoder(
                self._gen_cell,
                sampler,
                initial_state=initial_state,
                input_shape=start_inputs.shape[1:],
                output_layer=self._output_layer)

            final_output, final_state, final_lengths = seq2seq.dynamic_decode(
                decoder,
                maximum_iterations=max_length,
                swap_memory=True)
        return final_output.sample_id, final_lengths

    def pre_train(self, x_inputs, x_targets, x_length, max_length, z_size):
        """Pre-train generator

        Note:
        1）use-teacher forcing
        2）x_inputs is pad,
            [batch_size, time_step, pitch] ==> [batch_size, time_step, pitch+z_size]
        3）loss use cross_entropy

        :param x_inputs:
            [batch_size, time_step, pitch]
        :param x_targets:
            [batch_size, time_step, pitch]
        :param x_length:
            [batch_size, 1]
        :param max_length: a scalar number
        :param z_size: a scalar number
        :return: two operations about pre-train
            (pre_train_loss, pre_train_accuracy)
        """
        batch_size = x_inputs.shape[0]
        z_simulated = tf.zeros([batch_size, z_size], dtype=tf.float32)
        repeated_z = tf.tile(
            tf.expand_dims(z_simulated, axis=1), [1, tf.shape(x_inputs)[1], 1])
        x_inputs = tf.concat([x_inputs, repeated_z], axis=2)

        # Use teacher forcing.
        helper = seq2seq.TrainingHelper(x_inputs, x_length)

        with tf.variable_scope(self._name_or_scope, reuse=tf.AUTO_REUSE):
            initial_state = lstm_utils.initial_cell_state_from_embedding(self._gen_cell, z_simulated)

            decoder = lstm_utils.Seq2SeqLstmDecoder(
                self._gen_cell,
                helper,
                initial_state=initial_state,
                input_shape=x_inputs.shape[2:],
                output_layer=self._output_layer)

            final_output, _, _ = seq2seq.dynamic_decode(
                decoder,
                maximum_iterations=max_length,
                swap_memory=True)

            final_p_logits = final_output.rnn_output

            flat_x_targets = flatten_maybe_padded_sequences(x_targets, x_length)
            flat_p_logits = flatten_maybe_padded_sequences(final_p_logits, x_length)

            pre_train_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=flat_x_targets, logits=flat_p_logits)

            flat_truth = tf.argmax(flat_x_targets, axis=1)
            flat_predictions = tf.argmax(flat_p_logits, axis=1)
            pre_train_accuracy = tf.metrics.accuracy(flat_truth, flat_predictions)

        return pre_train_loss, pre_train_accuracy


# TODO 提供训练过程对 指定隐变量 的生成
