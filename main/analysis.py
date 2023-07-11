import numpy as np
import tensorflow as tf
from scipy import linalg
import hess


@tf.function
def flatten(grad):
    temp = tf.TensorArray(tf.float32, size=0,
                          dynamic_size=True, infer_shape=False)
    for gr in grad:
        temp = temp.write(temp.size(), tf.reshape(
            gr, (tf.math.reduce_prod(tf.shape(gr)), )))
    return temp.concat()


def load_weights(model, model_str):
    model.load_weights(model_str+'/saved_model/'+model_str)
    return model


def get_layer_pointer(model, layer_str):
    layer_name = []
    if layer_str == "all":
        layer_name = model.trainable_variables
    else:
        if layer_str[:6] == "layers":
            layer_name = [getattr(model, "layers")[int(
                layer_str[7:(len(layer_str)-1)])].kernel]
        else:
            layer_name = [getattr(model, layer_str).kernel]
    return layer_name


def get_weights(layer_pointer):
    return flatten(layer_pointer)


def load_evh(model_str, layer_str):
    return np.load(model_str+'/evh_'+layer_str+'.npy')


def weights_prod(weights, evh):
    return np.tensordot(weights/np.linalg.norm(weights), evh, axes=(0, 1))


def svd_conv(layer_pointer):
    tensor = np.array(layer_pointer)
    return tensor.reshape((np.prod(tensor.shape[0:2]),
                           np.prod(tensor.shape[2:4])))


def svd(matrix):
    U, s, Vh = linalg.svd(matrix)

    def s_weights(i):
        sing = np.zeros((U.shape[0], Vh.shape[0]))
        sing[i, i] = s[i]
        w_tilde = np.ndarray.flatten(np.matmul(U, np.matmul(sing, Vh)))
        w_tilde /= np.linalg.norm(w_tilde)
        return w_tilde
    svv = np.zeros((s.shape[0], np.prod(matrix.shape)))
    for i in range(s.shape[0]):
        svv[i] = s_weights(i)
    return svv


def evh_weights_prod(
        model,
        model_str,
        layer_str,
        x_train=None,
        y_train=None,
        batch_size=1000,
        tens=True):
    model = load_weights(model, model_str)
    layer_pointer = get_layer_pointer(model, layer_str)
    weights = get_weights(layer_pointer)
    if np.isin(None, x_train):
        evh = load_evh(model_str, layer_str)
    else:
        ewh, evh = hess.comp_eig(
            model, layer_str, x_train, y_train, batch_size=batch_size, tens=tens)
    return weights_prod(weights, evh)


def evh_weights_max(model, evh, layer_str):
    layer_pointer = get_layer_pointer(model, layer_str)
    weights = get_weights(layer_pointer)
    return np.argmax(np.abs(weights_prod(weights, evh)))


def sv_field(model, model_str, layer_str):
    model = load_weights(model, model_str)
    layer_pointer = get_layer_pointer(model, layer_str)[0]
    if len(layer_pointer.shape) > 2:
        matrix = svd_conv(layer_pointer)
    else:
        matrix = layer_pointer
    svv = svd(matrix)
    evh = load_evh(model_str, layer_str)
    return np.tensordot(svv, evh, axes=(1, 1))
