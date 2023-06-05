import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, regularizers, callbacks
import time
import argparse
from scipy import signal, stats, integrate, linalg,optimize, sparse
parser = argparse.ArgumentParser(
    prog='train MLP256',
    description='train MLP256',
    epilog='Based on the rmt package.')
parser.add_argument(
    '-n', '--name',
    type=str,
    default="missing name",
    help="Name of network",
    dest="name"
)
parser.add_argument(
    '-r', '--reg',
    type=float,
    default=0,
    help="Regularization",
    dest="reg"
)
parser.add_argument(
    '-nv', '--n_ev',
    type=int,
    default=0,
    help="Number of eigenvectors to compute",
    dest="n_ev"
)


args=parser.parse_args()
model_str=args.name
if args.reg!=0:
  reg=tf.keras.regularizers.l2(args.reg)
else:
  reg=None


n_eigenvectors=args.n_ev
exec(open(model_str+'/'+model_str+".py").read())
model.load_weights(model_str+'/saved_model/'+model_str)
# the list which contains all parameters we want to get the Hessian from (here no bias)
layer_name=[]
for layer in model.trainable_weights:
  if len(layer.shape)>1:
    layer_name.append(layer)
#computes the number of parameters of the layer
layer_size = 0
for layer in layer_name:
    j = 1
    for i in layer.shape:
        j *= i
    layer_size += j
print(layer_size)
#conserve the numpy datatype
layer_dtype = np.float32
layer_lengths=[]
for layer in layer_name:
  layer_lengths.append(tf.math.reduce_prod(layer.shape))
# returns hessian vector product of given list or tensor
@tf.function
def hessian_calc(vector):
    with tf.GradientTape(watch_accessed_variables=False) as tape1:
        tape1.watch(layer_name)
        with tf.GradientTape(watch_accessed_variables=False) as tape2:
            tape2.watch(layer_name)
            # No trainig -> batch norm
            predictions = model(x_train, training=False)
            loss = loss_object(y_train, predictions)
            loss+=sum(model.losses)
        gradient=tape2.gradient(loss, layer_name)
    return tape1.gradient(gradient,layer_name,output_gradients=vector)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
# old packing function which works fine, but slower? 
"""@tf.function
def repack(weights):
    weight_split=tf.split(weights,layer_lengths)
    templer = []
    for layer,weights_set in zip(layer_name,weight_split):
        templ = tf.reshape(weights_set,layer.shape)
        templer.append(templ)
    return templer"""
# packs the flatten parameter back to a list and in right shape
@tf.function
def repack(weights):
    weight_split=tf.split(weights,layer_lengths)
    templer = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
    for layer,weights_set in zip(layer_name,weight_split):
        templer.write(templer.size(),tf.reshape(weights_set,layer.shape))
    return templer.stack()
# flattens list to vector
@tf.function
def flatten(grad):
    temp = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False)
    for gr in grad:
        temp = temp.write(temp.size(), tf.reshape(gr, (tf.math.reduce_prod(tf.shape(gr)), )))
    return temp.concat()
@tf.function
def hvp(vector):
  # prints time in error log
  tf.print(tf.timestamp())
  return flatten(hessian_calc(repack(vector)))
linOP=sparse.linalg.LinearOperator((layer_size,layer_size),matvec=hvp,dtype=layer_dtype)
ew,ev=sparse.linalg.eigsh(linOP,n_eigenvectors)
np.save(model_str+'/ewh_sparse',ew[::-1])
np.save(model_str+'/evh_sparse',ev.T[::-1])

