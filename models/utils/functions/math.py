import tensorflow as tf


def reduce_sum(
    input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None
):
    try:
        return tf.reduce_sum(
            input_tensor,
            axis=axis,
            keep_dims=keep_dims,
            name=name,
            reduction_indices=reduction_indices,
        )
    except TypeError:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keep_dims, name=name)


def reduce_max(
    input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None
):
    try:
        return tf.reduce_max(
            input_tensor,
            axis=axis,
            keep_dims=keep_dims,
            name=name,
            reduction_indices=reduction_indices,
        )
    except TypeError:
        return tf.reduce_max(input_tensor, axis=axis, keepdims=keep_dims, name=name)


def div(x, y, name=None):
    try:
        return tf.div(x, y, name=name)
    except AttributeError:
        return tf.divide(x, y, name=name)


def softmax(logits, dim=-1, name=None):
    try:
        return tf.nn.softmax(logits, dim=dim, name=name)
    except TypeError:
        return tf.nn.softmax(logits, axis=dim, name=name)
