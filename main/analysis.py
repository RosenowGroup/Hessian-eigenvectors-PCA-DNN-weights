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


@tf.function
def repack(weights, layer_pointer, layer_lengths):
    weight_split = tf.split(weights, layer_lengths)
    templer = []
    for layer, weights_set in zip(layer_pointer, weight_split):
        templ = tf.reshape(weights_set, layer.shape)
        templer.append(templ)
    return templer


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


def get_layer_lengths(layer_pointer):
    layer_lengths = []
    for layer in layer_pointer:
        layer_lengths.append(tf.math.reduce_prod(layer.shape))
    return layer_lengths


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
            model, layer_str, x_train, y_train,
            batch_size=batch_size, tens=tens)
    return weights_prod(weights, evh)


def evh_weights_max(model, evh, layer_str):
    layer_pointer = get_layer_pointer(model, layer_str)
    weights = get_weights(layer_pointer)
    return np.argmax(np.abs(weights_prod(weights, evh)))


def sv_field(model, model_str, layer_str, md_str="evh", layer_index=-1):
    model = load_weights(model, model_str)
    if layer_str == 'all':
        layer_pointer = model.layers[layer_index].kernel
        if len(layer_pointer.shape) > 2:
            matrix = svd_conv(layer_pointer)
        else:
            matrix = layer_pointer
    else:
        layer_pointer = get_layer_pointer(model, layer_str)[0]
        if len(layer_pointer.shape) > 2:
            matrix = svd_conv(layer_pointer)
        else:
            matrix = layer_pointer
    svv = svd(matrix)
    if layer_str == 'all':
        i_start = 0
        for layer in model.trainable_variables[:layer_index]:
            i_start += np.prod(layer.shape)
        layer_size = np.sum(get_layer_lengths(layer_pointer))
        evh = load_evh(model_str, "all")[i_start:i_start+layer_size]
    else:
        evh = load_evh(model_str, layer_str)
    evh = np.load(model_str+'/'+md_str+'_'+layer_str+'.npy')
    return np.tensordot(svv, evh, axes=(1, 1))


def acc_components(
        model,
        layer_str,
        vecs,
        x_train,
        y_train,
        x_test,
        y_test,
        batch_size=1000):
    layer_name = []
    b_list = []
    if layer_str == "all":
        layer_name = model.trainable_variables
    else:
        if layer_str[:6] == "layers":
            layer_name = [getattr(model, "layers")[int(
                layer_str[7:(len(layer_str)-1)])].kernel]
            b_list = [getattr(model, "layers")[int(
                layer_str[7:(len(layer_str)-1)])].bias]
        else:
            layer_name = [getattr(model, layer_str).kernel]
            b_list = [getattr(model, layer_str).bias]
    layer_lengths = get_layer_lengths(layer_name)
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).batch(batch_size)

    weights_0 = flatten(layer_name)

    theta = np.tensordot(weights_0, vecs, axes=(0, 1))
    if layer_str == "all":
        def set_weights(weights):
            templ = repack(weights, layer_name, layer_lengths)
            k = 0
            for i in range(len(model.layers)):
                if len(model.layers[i].get_weights()) == 2:
                    model.layers[i].set_weights([templ[k], templ[k+1]])
                    k += 2
                elif len(model.layers[i].get_weights()) == 1:
                    model.layers[i].set_weights([templ[k]])
                    k += 1
    else:
        def set_weights(weights):
            getattr(model, layer_str).set_weights(
                [repack(weights, layer_name, layer_lengths)[0], b_list[0]])

    # adding the principal components together up to ith indices
    def sum_weights(proj, vec):
        return np.tensordot(proj, vec, axes=[0, 0])

    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    @tf.function
    def calc_acc(images, labels):
        predictions = model(images, training=False)
        test_accuracy(labels, predictions)
        loss = loss_object(labels, predictions)
        loss += sum(model.losses)
        train_loss(loss)
    step_range = vecs.shape[0]
    step_array = np.arane(step_range)
    testacc = np.empty(step_array.shape[0])
    trainacc = np.empty(step_array.shape[0])
    testloss = np.empty(step_array.shape[0])
    trainloss = np.empty(step_array.shape[0])
    index = 0

    for i in step_array:
        set_weights(sum_weights(theta[:i], vecs[:i]))
        for images, labels in train_ds:
            calc_acc(images, labels)
        trainacc[index] = test_accuracy.result()
        trainloss[index] = train_loss.result()
        train_loss.reset_states()
        test_accuracy.reset_states()
        calc_acc(x_test, y_test)
        testacc[index] = test_accuracy.result()
        testloss[index] = train_loss.result()
        train_loss.reset_states()
        test_accuracy.reset_states()
        index += 1
    return np.array([step_array, trainacc, testacc, trainloss, testloss])


def loss_landscape(model, layer_str, vec, epsilons, x_train, y_train, x_test, y_test, batch_size=1000):
    layer_name = []
    b_list = []
    if layer_str == "all":
        for layer in model.trainable_weights:
            if len(layer.shape) == 1:
                b_list.append(np.array(layer))
            if len(layer.shape) > 0:
                layer_name.append(np.array(layer))
    else:
        for layer in getattr(model, layer_str).get_weights():
            if len(layer.shape) == 1:
                b_list.append(layer)
            if len(layer.shape) > 1:
                layer_name.append(layer)
    weights_0 = flatten(layer_name)
    layer_lengths = get_layer_lengths(layer_name)
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).batch(batch_size)

    if layer_str == "all":
        def set_weights(weights):
            templ = repack(weights, layer_name, layer_lengths)
            k = 0
            for i in range(len(model.layers)):
                if len(model.layers[i].get_weights()) == 2:
                    model.layers[i].set_weights([templ[k], templ[k+1]])
                    k += 2
                elif len(model.layers[i].get_weights()) == 1:
                    model.layers[i].set_weights([templ[k]])
                    k += 1
    else:
        def set_weights(weights):
            templ = repack(weights, layer_name, layer_lengths)
            getattr(model, layer_str).set_weights([templ[0], b_list[0]])

    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='test_accuracy')
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    @tf.function
    def calc_acc(images, labels):
        predictions = model(images, training=False)
        test_accuracy(labels, predictions)
        loss = loss_object(labels, predictions)
        loss += sum(model.losses)
        train_loss(loss)

    trainacc = np.empty(epsilons.shape[0])
    trainloss = np.empty(epsilons.shape[0])
    testacc = np.empty(epsilons.shape[0])
    index = 0
    for epsilon in epsilons:
        set_weights(weights_0+epsilon*vec)
        for images, labels in train_ds:
            calc_acc(images, labels)
        trainacc[index] = test_accuracy.result()

        trainloss[index] = train_loss.result()
        train_loss.reset_states()
        test_accuracy.reset_states()
        calc_acc(x_test, y_test)
        testacc[index] = test_accuracy.result()
        train_loss.reset_states()
        test_accuracy.reset_states()
        index += 1
    return np.array([epsilons, trainacc, trainloss, testacc])
