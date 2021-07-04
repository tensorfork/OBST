import typing

import tensorflow as tf


def scoped(name: str, fn: typing.Callable, *args, **kwargs):
    return scoped(f"tf_{name}", fn, *args, **kwargs)


def softplus(tensor: tf.Tensor) -> tf.Tensor:
    return scoped("softplus", tf.math.softplus, tensor)


def divide(x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
    return scoped("divide", tf.math.divide, x1, x2)


def multiply(x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
    return scoped("multiply", tf.math.multiply, x1, x2)


def add(x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
    return scoped("add", tf.math.add, x1, x2)


def subtract(x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
    return scoped("subtract", tf.math.subtract, x1, x2)


def tanh(tensor: tf.Tensor) -> tf.Tensor:
    return scoped("divide", tf.math.tanh, tensor)


def square(tensor: tf.Tensor) -> tf.Tensor:
    return scoped("divide", tf.math.square, tensor)


def sigmoid(tensor: tf.Tensor) -> tf.Tensor:
    return scoped("sigmoid", tf.math.sigmoid, tensor)


def abs(tensor: tf.Tensor) -> tf.Tensor:
    return scoped("abs", tf.math.abs, tensor)
