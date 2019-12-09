# coding=utf-8

"""
This file defines common GAN losses.

source: https://github.com/salu133445/musegan
"""
import tensorflow as tf


def get_loss_fn(kind):
    """Return the related GAN's loss function."""
    if kind == 'classic':
        loss_fn = classic_gan_losses
    elif kind == 'nonsaturating':
        loss_fn = nonsaturating_gan_losses
    elif kind == 'wasserstein':
        loss_fn = wasserstein_gan_losses
    elif kind == 'hinge':
        loss_fn = hinge_gan_losses
    return loss_fn


def get_adv_losses(discriminator_real_outputs, discriminator_fake_outputs,
                   kind):
    """Return the corresponding GAN losses for the generator and the
    discriminator."""
    if kind == 'classic':
        loss_fn = classic_gan_losses
    elif kind == 'nonsaturating':
        loss_fn = nonsaturating_gan_losses
    elif kind == 'wasserstein':
        loss_fn = wasserstein_gan_losses
    elif kind == 'hinge':
        loss_fn = hinge_gan_losses
    return loss_fn(discriminator_real_outputs, discriminator_fake_outputs)


def classic_gan_losses(discriminator_real_outputs, discriminator_fake_outputs):
    """Return the classic GAN losses for the generator and the discriminator.

    (Generator)      log(1 - sigmoid(D(G(z))))
    (Discriminator)  - log(sigmoid(D(x))) - log(1 - sigmoid(D(G(z))))
    """
    discriminator_loss_real = tf.losses.sigmoid_cross_entropy(
        tf.ones_like(discriminator_real_outputs), discriminator_real_outputs)
    discriminator_loss_fake = tf.losses.sigmoid_cross_entropy(
        tf.zeros_like(discriminator_fake_outputs), discriminator_fake_outputs)
    discriminator_loss = discriminator_loss_real + discriminator_loss_fake
    generator_loss = -discriminator_loss_fake
    return generator_loss, discriminator_loss


def nonsaturating_gan_losses(discriminator_real_outputs,
                             discriminator_fake_outputs):
    """Return the non-saturating GAN losses for the generator and the
    discriminator.

    (Generator)      -log(sigmoid(D(G(z))))
    (Discriminator)  -log(sigmoid(D(x))) - log(1 - sigmoid(D(G(z))))
    """
    discriminator_loss_real = tf.losses.sigmoid_cross_entropy(
        tf.ones_like(discriminator_real_outputs), discriminator_real_outputs)
    discriminator_loss_fake = tf.losses.sigmoid_cross_entropy(
        tf.zeros_like(discriminator_fake_outputs), discriminator_fake_outputs)
    discriminator_loss = discriminator_loss_real + discriminator_loss_fake
    generator_loss = tf.losses.sigmoid_cross_entropy(
        tf.ones_like(discriminator_fake_outputs), discriminator_fake_outputs)
    return generator_loss, discriminator_loss


def wasserstein_gan_losses(discriminator_real_outputs,
                           discriminator_fake_outputs):
    """Return the Wasserstein GAN losses for the generator and the
    discriminator.

    (Generator)      -D(G(z))
    (Discriminator)  D(G(z)) - D(x)
    """
    generator_loss = -tf.reduce_mean(discriminator_fake_outputs)
    discriminator_loss = -generator_loss - tf.reduce_mean(
        discriminator_real_outputs)
    return generator_loss, discriminator_loss


def hinge_gan_losses(discriminator_real_outputs, discriminator_fake_outputs):
    """Return the Hinge GAN losses for the generator and the discriminator.

    (Generator)      -D(G(z))
    (Discriminator)  max(0, 1 - D(x)) + max(0, 1 + D(G(z)))
    """
    generator_loss = -tf.reduce_mean(discriminator_fake_outputs)
    discriminator_loss = (
            tf.reduce_mean(tf.nn.relu(1. - discriminator_real_outputs))
            + tf.reduce_mean(tf.nn.relu(1. + discriminator_fake_outputs)))
    return generator_loss, discriminator_loss


def wasserstein_d_update(loss, optimizer, clipping_value, var_list=None, name='d_update'):
    """Used to update discriminator's parameters

    :param loss: discriminator's loss function
    :param optimizer: specify which optimizer to use, usually RMSProp
    :param clipping_value: parameters clipped to [-clipping_value, clipping_value], after gradient descent
    :param var_list
    :param name: specify op name
    :return: a tensor op
    """
    # gradients, var_list = zip(*optimizer.compute_gradients(loss, var_list=var_list))
    # optimizer.apply_gradients(zip(gradients, var_list), name=name)
    # return clip_discriminator_var_op

    opt_op = optimizer.minimize(loss, var_list=var_list, name=name)
    with tf.control_dependencies([opt_op]):
        clip_discriminator_var_op = [
            var.assign(tf.clip_by_value(var, -clipping_value, clipping_value))
            for var in var_list]
    return clip_discriminator_var_op


def wasserstein_g_update(loss, optimizer, var_list=None, global_step=None, name='g_update'):
    """Used to update generator's parameters

    :param loss: generator's loss function
    :param optimizer: specify which optimizer to use, usually RMSProp
    :param var_list:
    :param global_step: this's be added 1, after update
    :param name: specify op name
    :return: a tensor op
    """
    # grads = optimizer.compute_gradients(loss, var_list=var_list)
    # return optimizer.apply_gradients(grads, global_step=global_step, name=name)

    return optimizer.minimize(loss, global_step=global_step, var_list=var_list, name=name)
