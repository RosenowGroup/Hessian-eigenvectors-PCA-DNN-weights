
# Commented out IPython magic to ensure Python compatibility.
"""
Lanczos Algorithm to compute the "most important" eigenvectors and eigenvalues
of the hessian of possibly any dnn
"""
# import dependencies

import numpy as np
import tensorflow as tf
import time
from tensorflow.keras import layers


# load the data set
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train/255, x_test/255
# normalise the data
for i in range(x_train.shape[0]):
    x_train[i] -= np.mean(x_train[i], axis=(0, 1))
    x_train[i] /= np.std(x_train[i], axis=(0, 1))

# network initalization
model_str = 'fc_256_32urc'
reg = None  # tf.keras.regularizers.l2(5e-4)


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(256, activation='relu', kernel_regularizer=reg)
        self.d2 = layers.Dense(128, activation='relu', kernel_regularizer=reg)
        self.d3 = layers.Dense(64, activation='relu', kernel_regularizer=reg)
        self.d4 = layers.Dense(10, activation='softmax',
                               kernel_regularizer=reg)

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return self.d4(x)


model = MyModel()
model.build((32, 3072))
# load network from data to avoid compatibility problems
model.load_weights(model_str+'/saved_model/'+model_str)

# list of layers from which we want to compute the eigenvectors
layer_name = []
for layer in model.trainable_weights:
    if len(layer.shape) == 2:
        layer_name.append(layer)
# computes the number of parameters of the layer
layer_size = 0
for layer in layer_name:
    j = 1
    for i in layer.shape:
        j *= i
    layer_size += j

layer_dtype = np.float32
# lengths of corresponding layers
layer_lengths = []
for layer in layer_name:
    layer_lengths.append(tf.math.reduce_prod(layer.shape))


@tf.function
def hessian_calc(images, labels, vector):
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
            # term to include regularization
            loss += sum(model.losses)
        gradient = tape2.gradient(loss, layer_name)
    return tape1.gradient(gradient, layer_name, output_gradients=vector)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy()


@tf.function
def repack(weights):

    weight_split = tf.split(weights, layer_lengths)
    templer = []
    i_last = 0
    for layer in layer_name:
        templ = tf.reshape(weight_split[i_last], layer.shape)
        i_last += 1
        templer.append(templ)
    return templer


@tf.function
def flatten(grad):
    temp = tf.TensorArray(tf.float32, size=0,
                          dynamic_size=True, infer_shape=False)
    for gr in grad:
        temp = temp.write(temp.size(), tf.reshape(
            gr, (tf.math.reduce_prod(tf.shape(gr)), )))
    return temp.concat()

# draw orthonormal vector for lanczos


@tf.function
def get_orth(V):
    rand = g.normal(shape=V[0].shape)
    ortho = rand*1
    for vector in V:
        ortho -= tf.tensordot(rand, vector, axes=1)*vector
    temp = tf.linalg.norm(ortho)
    # ensures we never get a zero vector (unstable)
    if temp == 0:
        return get_orth(V)
    return ortho/temp


def lanczos(vector: tf.Tensor, m: int):
    time_stemp = time.monotonic()
    time_stemp_0 = time_stemp
    # diagonal elements
    alpha = []
    # subdiagonal elements
    beta = []
    # vectors of the lanczos alogrithm
    V = []
    vector_norm = vector/tf.linalg.norm(vector)
    w = flatten(hessian_calc(x_train, y_train, repack(vector_norm)))
    a = tf.tensordot(w, vector_norm, axes=1)
    w = w-a*vector_norm
    alpha.append(a)
    V.append(vector_norm)
    for i in tf.range(m-1):
        print(str(int(i))+' of '+str(m-2))
        b = tf.linalg.norm(w)
        vector_norm = w/b
        if b == 0:
            vector_norm = get_orth(V)
        w = flatten(hessian_calc(x_train, y_train, repack(vector_norm)))
        a = tf.tensordot(w, vector_norm, axes=1)
        w = w-a*vector_norm-b*V[-1]
        alpha.append(a)
        beta.append(b)
        V.append(vector_norm)
        time_temp = time.monotonic()
        time_diff = time_temp - time_stemp
        time_stemp = time_temp
        time_total = time_temp - time_stemp_0
        time_rem = (time_total/(int(i)+1))*(m-int(i)-2)
        print(
            f'{time_diff:.1f}s, '
            f'{time_rem:.1f}s'
        )
    # solve the tridiagonal eigenvalue problem
    ew, ev = tf.linalg.eigh_tridiagonal(alpha, beta, eigvals_only=False)
    # convert the eigenvectors to the space of the Hessian
    x = tf.tensordot(V, ev, axes=[[0], [0]])
    return ew[::-1], tf.transpose(x)[::-1]


# number of eigenvectors that are supposed to be computed
n_eigenvectors = 6
# create a random vector to start with the algorithm
g = tf.random.Generator.from_seed(1234)
rand = g.normal(shape=[layer_size])
# output eigenvalues, eigenvectors
ew, ev = lanczos(rand, n_eigenvectors)
