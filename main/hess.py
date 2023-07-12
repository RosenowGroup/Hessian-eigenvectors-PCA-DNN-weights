import time

import numpy as np
import tensorflow as tf
from scipy import sparse


def hess(model, layer_str, x_train, y_train, batch_size=1000):
    """
    Computes the Hessian matrix of a tensorflow model.

    Args:
      model: model.
      layer_str: str of a layer, i.e. "layers[2]" for layer 3,
        or "all" for all layers.
      x_train: training samples.
      y_train: training labels.
      batch_size: batch size of training samples, must be set small enough
        such that the GPU does not run out of memory.

    Returns:
      The Hessian matrix.
    """
    hess_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).batch(batch_size)
    n_batches = hess_ds.cardinality().numpy()
    if layer_str == "all":
        layer_name = model.trainable_weights
    else:
        if layer_str[:6] == "layers":
            layer_name = [getattr(model, "layers")[int(
                layer_str[7:(len(layer_str)-1)])].kernel]
        else:
            layer_name = [getattr(model, layer_str).kernel]
    layer_lengths = []
    for layer in layer_name:
        layer_lengths.append(tf.math.reduce_prod(layer.shape))
    layer_size = 0
    for layer in layer_name:
        layer_size += np.prod(layer.shape)
    # conserve the numpy datatype
    layer_dtype = np.float32

    @tf.function
    def hessian_calc(vector, images, labels):
        with tf.GradientTape(watch_accessed_variables=False) as tape1:
            tape1.watch(layer_name)
            with tf.GradientTape(watch_accessed_variables=False) as tape2:
                tape2.watch(layer_name)
                # No trainig -> batch norm
                predictions = model(images, training=False)
                loss = loss_object(labels, predictions)
                loss += sum(model.losses)
            gradient = tape2.gradient(loss, layer_name)
        return tape1.gradient(gradient, layer_name, output_gradients=vector)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    @tf.function
    def repack(weights):
        # may be improved, since it can not be compiled
        weight_split = tf.split(weights, layer_lengths)
        templer = []
        for layer, weights_set in zip(layer_name, weight_split):
            templ = tf.reshape(weights_set, layer.shape)
            templer.append(templ)
        return templer

    @tf.function
    def flatten(grad):
        temp = tf.TensorArray(tf.float32, size=0,
                              dynamic_size=True, infer_shape=False)
        for g in grad:
            temp = temp.write(temp.size(), tf.reshape(
                g, (tf.math.reduce_prod(tf.shape(g)), )))
        return temp.concat()

    def comp_hess():
        time_stemp = time.monotonic()
        time_stemp_0 = time_stemp
        vector = np.zeros(layer_size, dtype=layer_dtype)
        hess = np.zeros((layer_size, layer_size), dtype=layer_dtype)
        for i in range(layer_size):
            vector[i] = 1
            packed = repack(vector)
            for images, labels in hess_ds:
                hess[i] += flatten(hessian_calc(packed, images, labels))
            vector[i] = 0
            time_temp = time.monotonic()
            time_diff = time_temp - time_stemp
            time_stemp = time_temp
            time_total = time_temp - time_stemp_0
            time_rem = (time_total/(i+1))*(layer_size-i-1)
            print(
                f'{time_diff:.1f}s, '
                f'{time_rem:.1f}s'
            )
        return hess/n_batches
    return comp_hess()


def eigh(hess, tens=True):
    """
    Computes the eigenvalues and eigenvectors
      of the Hessian or a hermitian matrix in general.

    Args:
      hess: Hessian matrix or matrix.
      tens: bool that determines, if the eigensolver
        of numpy or tensorflow should be used
        defaults to True, the tensorflow eigensolver.

    Returns:
      eigenvalues [i], eigenvectors [i,:] in decreasing order as numpy arrays.
    """
    if tens:
        eigh, vech = tf.linalg.eigh(hess)
    else:
        eigh, vech = np.linalg.eigh(hess)
    return np.array(eigh)[::-1], np.transpose(vech)[::-1]


def comp_eig(model, layer_str, x_train, y_train, batch_size=1000, tens=True):
    """
    Computes the eigenvalues and eigenvectors of model directly.

    Args:
        model: model.
        layer_str: str of a layer, i.e. "layers[2]" for layer 3,
          or "all" for all layers.
        x_train: training samples.
        y_train: training labels.
        batch_size: batch size of training samples, must be set small enough
          such that the GPU does not run out of memory.
        tens: bool that determines, if the eigensolver
          of numpy or tensorflow should be used
          defaults to True, the tensorflow eigensolver.

    Returns:
      eigenvalues [i], eigenvectors [i,:] in decreasing order as numpy arrays.
    """
    return eigh(hess(model, layer_str, x_train, y_train, batch_size), tens)


def lancz_single(model, layer_pointer, number_cons, images, labels):
    """
    Computes the most important eigenvalues
      and eigenvectors of a layer of a model with the Lanczos algorithm.

    Args:
        model: model.
        layer_pointer: tensor objects that points to the parameters
          of the model, from which one wants.
        number_cons: number of eigenvalues to compute.
        images: training samples.
        labels: training labels.

    Returns:
      eigenvalues [i], eigenvectors [i,:] in decreasing order as numpy arrays.
    """
    layer_size = np.prod(layer_pointer.shape)

    @tf.function
    def hessian_calc(vector):
        with tf.GradientTape(watch_accessed_variables=False) as tape1:
            tape1.watch(layer_pointer)
            with tf.GradientTape(watch_accessed_variables=False) as tape2:
                tape2.watch(layer_pointer)
                predictions = model(images, training=False)
                loss = loss_object(labels, predictions)
                loss += sum(model.losses)
            gradient = tape2.gradient(loss, layer_pointer)
        return tape1.gradient(gradient, layer_pointer, output_gradients=vector)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    @tf.function
    def hvp(vector):
        return hessian_calc(tf.reshape(vector, layer_pointer.shape))
    linOP = sparse.linalg.LinearOperator(
        (layer_size, layer_size), matvec=hvp, dtype=np.float32)
    ew, ev = sparse.linalg.eigsh(linOP, number_cons)
    return ew[::-1], ev.T[::-1]
