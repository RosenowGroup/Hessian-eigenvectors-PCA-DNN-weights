import numpy as np
import tensorflow as tf
from scipy import linalg
import hess
from scipy.special import erf


@tf.function
def get_layer_lengths(layer_pointer_list):
    """
    Computes the lengths of tensors of a list.

    Args:
      layer_pointer_list:  list of tensors.

    Returns:
      List with lengths of tensors.
    """
    layer_lengths = []
    for layer in layer_pointer_list:
        layer_lengths.append(tf.math.reduce_prod(layer.shape))
    return layer_lengths


@tf.function
def flatten(grad):
    """
    Flattens list of tensors to a vector.

    Args:
      grad:  list of tensors.

    Returns:
      Vector that contains all entries of the list.
    """
    temp = tf.TensorArray(tf.float32, size=0,
                          dynamic_size=True, infer_shape=False)
    for gr in grad:
        temp = temp.write(temp.size(), tf.reshape(
            gr, (tf.math.reduce_prod(tf.shape(gr)), )))
    return temp.concat()


@tf.function
def repack(vec, layer_pointer_list):
    """
    Repacks the list of tensors from a vector.

    Args:
      vec:  vector.
      layer_pointer_list: list of tensors.

    Returns:
      List of tensors as used in TensorFlow.
    """
    layer_lengths = get_layer_lengths(layer_pointer_list)
    weight_split = tf.split(vec, layer_lengths)
    templer = []
    for layer, weights_set in zip(layer_pointer_list, weight_split):
        templ = tf.reshape(weights_set, layer.shape)
        templer.append(templ)
    return templer


def get_layer_pointer(model, layer_str):
    """
    Extract the layer pointer from a model.

    Args:
      model:  model object.
      layer_str: str of a layer, i.e. "layers[2]" for layer 3,
        or "all" for all layers.

    Returns:
      A layer pointer.
    """
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


def load_weights(model, model_str):
    """
    Loads the weights of a model.

    Args:
      model:  model object.
      model_str: name of the model, s.t. its folder can be accessed.

    Returns:
      Model with weights loaded.
    """
    model.load_weights(model_str+'/saved_model/'+model_str)
    return model


def load_evh(model_str, layer_str):
    """
    Loads the Hessian eigenvectors.

    Args:
      model_str: name of the model, s.t. its folder can be accessed.
      layer_str: name of the layer, s.t. its folder can be accessed.

    Returns:
      Numpy array of eigenvectors.
    """
    return np.load(model_str+'/evh_'+layer_str+'.npy')


def weights_prod(weights, evh):
    """
    Computes the scalar product between the weights and the eigenvectors.

    Args:
      weights: weights as an vector.
      evh: eigenvectors.

    Returns:
      Numpy array of the weights product.
    """
    return np.tensordot(weights/np.linalg.norm(weights), evh, axes=(0, 1))


def svd_conv(layer_pointer):
    """
    Reshapes a convolutional layer to a matrix.

    Args:
      layer_pointer: Tensor or array of the convolutional layer in tensor form.

    Returns:
      Numpy array of the matrix.
    """
    tensor = np.array(layer_pointer)
    return tensor.reshape((np.prod(tensor.shape[0:2]),
                           np.prod(tensor.shape[2:4])))


def svd(matrix):
    """
    Computes the extended singular vectors.

    Args:
      matrix: Matrix where the SVD should be taken from.

    Returns:
      Extended singular vectors [i,:] in decreasing order.
    """
    U, s, Vh = linalg.svd(matrix)

    def s_weights(i):
        sing = np.zeros((U.shape[0], Vh.shape[0]), dtype=np.float32)
        sing[i, i] = s[i]
        w_tilde = np.ndarray.flatten(np.matmul(U, np.matmul(sing, Vh)))
        w_tilde /= np.linalg.norm(w_tilde)
        return w_tilde
    svv = np.zeros((s.shape[0], np.prod(matrix.shape)), dtype=np.float32)
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
    """
    Computes the weights product directly from a trained model.

    Args:
      model: model object.
      model_str: name of the model, s.t. its folder can be accessed.
      layer_str: str of a layer, i.e. "layers[2]" for layer 3,
        or "all" for all layers.
      x_train: optional, training samples,
        if given the Hessian eigenvectors are computed.
      y_train: optional, training labels.
      batch_size: optional, batch size
        for the computation of Hessian eigenvectors.
      tens: optional, bool that determines, if the eigensolver
          of numpy or tensorflow should be used
          defaults to True, the tensorflow eigensolver.

    Returns:
      Extended singular vectors [i,:] in decreasing order.
    """
    model = load_weights(model, model_str)
    layer_pointer = get_layer_pointer(model, layer_str)
    weights = flatten(layer_pointer)
    if np.isin(None, x_train):
        evh = load_evh(model_str, layer_str)
    else:
        ewh, evh = hess.comp_eig(
            model, layer_str, x_train, y_train,
            batch_size=batch_size, tens=tens)
    return weights_prod(weights, evh)


def sv_field(model, model_str, layer_str, md_str="evh", layer_index=-1):
    """
    Computes the extended singular vector and eigenbasis field.

    Args:
      model: model object.
      model_str: name of the model, s.t. its folder can be accessed.
      layer_str: str of a layer for the eigenbasis
        or "all" for all layers.
      md_str: optional, name of the eigenbasis, defaults to "evh".
      layer_index: required for layer_str="all",
        index of layer of where to compute the esv.

    Returns:
      Field [i,j], where i is the esv and j the eigenbasis index in decreasing order.
    """
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
        evh = np.load(model_str+'/'+md_str+'_'+layer_str +
                      '.npy')[i_start:i_start+layer_size]
    else:
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
    """
    Computes the accuracy when adding the eigenbasis decomposition
      starting from the first vec in vecs.

    Args:
      model: model object.
      layer_str: str of a layer for the eigenbasis
        or "all" for all layers.
      vecs: eigenbasis [i,:] is the ith vector.
      x_train: training samples.
      y_train: training labels.
      x_test: test samples.
      y_test: test labels.
      batch_size: optional, batch size.

    Returns:
      Array [N_add,training acc, test acc, training loss, test loss].
    """
    layer_name = []
    b_list = []
    if layer_str == "all":
        layer_name = model.trainable_variables
    else:
        if layer_str[:6] == "layers":
            layer_index = int(
                layer_str[7:(len(layer_str)-1)])
            layer_name = [getattr(model, "layers")[layer_index].kernel]
            b_list = [getattr(model, "layers")[layer_index].bias]
        else:
            layer_name = [getattr(model, layer_str).kernel]
            b_list = [getattr(model, layer_str).bias]
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).batch(batch_size)

    weights_0 = flatten(layer_name)

    theta = np.tensordot(weights_0, vecs, axes=(0, 1))
    if layer_str == "all":
        def set_weights(weights):
            templ = repack(weights, layer_name)
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
            if layer_str[:6] == "layers":
                getattr(model, "layers")[layer_index].set_weights(
                    [repack(weights, layer_name)[0], b_list[0]])
            else:
                getattr(model, layer_str).set_weights(
                    [repack(weights, layer_name)[0], b_list[0]])

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
    step_array = np.arange(step_range)
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


def loss_landscape(model,
                   layer_str,
                   vec,
                   epsilons,
                   x_train,
                   y_train,
                   x_test,
                   y_test,
                   batch_size=1000):
    """
    Computes the loss landscape in direction of a given vector.

    Args:
      model: model object.
      layer_str: str of a layer
        or "all" for all layers.
      vec: vector of which wants the landscape.
      epsilons: steps in direction of the vector.
      x_train: training samples.
      y_train: training labels.
      x_test: test samples.
      y_test: test labels.
      batch_size: optional, batch size.

    Returns:
      Array [epsilons,training acc, training loss, test acc].
    """
    layer_name = []
    b_list = []
    if layer_str == "all":
        for layer in model.trainable_weights:
            if len(layer.shape) == 1:
                b_list.append(np.array(layer))
            if len(layer.shape) > 0:
                layer_name.append(np.array(layer))
    else:
        if layer_str[:6] == "layers":
            layer_index = int(
                layer_str[7:(len(layer_str)-1)])
            layer_name = [getattr(model, "layers")[layer_index].kernel]
            b_list = [getattr(model, "layers")[layer_index].bias]
        else:
            layer_name = [getattr(model, layer_str).kernel]
            b_list = [getattr(model, layer_str).bias]
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
            if layer_str[:6] == "layers":
                getattr(model, "layers")[layer_index].set_weights(
                    [repack(weights, layer_name)[0], b_list[0]])
            else:
                getattr(model, layer_str).set_weights(
                    [repack(weights, layer_name)[0], b_list[0]])

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


def wigner(singVal, average=15):
    """
    Code taken from Max Staats.
    Wigner surmise.

    Args:
      singVal: singular values as an array.
      average: optional, number of singular values
        over which to average per side.

    Returns:
      Array of the unfolded surmise.
    """
    """def wigner( x ):
        return np.pi*x/2*np.exp( - np.pi*x**2 /4 )

    # cumulated wigner surmise
    def cum_wigner( x):
        return 1- np.exp( -x*x*np.pi/4)

    # cumulated distr
    def cumulated_dist( ev ):
        cum_distr= np.empty( len(ev) )
        for i in range( len(ev) ):
            cum_distr[i] = (i+1)/ len(ev)
        return cum_distr"""

    def int_unfolded_prob(array, average, x):
        result = 0
        for i in range(average, array.size - average):
            std = (array[i+average] - array[i-average])/2
            result += 0.5*(1+erf((x-array[i]) / (std * np.sqrt(2))))
        return result

    def get_surmise(singVal, average):
        unfolded_EV = np.empty(len(singVal)-4*average)
        for i in range(2*average, len(singVal)-2*average):
            unfolded_EV[i-2 *
                        average] = int_unfolded_prob(singVal,
                                                     average, singVal[i])
        spacings = np.diff(unfolded_EV)

        spacings = np.sort(spacings)

        return spacings
    return get_surmise(singVal, average)


def ptd(evh, number_test_s=10000, averaging_window=15):
    """
    Code taken from Max Staats.
    KS-test for the Porter Thomas distribution.

    Args:
      evh: eigenvectors on which to test the statistic.
      number_test_s: optional, number of examples
        drawn to simulate the statistic.
      averaging_window: optional, number of singular values
        over which to average per side.

    Returns:
      Array of the KS-test results for all eigenvectors.
    """
    tested_vec = evh.T.shape[0]

    def get_cdf(N):
        y_values = np.empty(N)
        for i in range(N):
            y_values[i] = (i+1)/N
        return y_values

    def get_N_normalised_vec(vector_l, number_vec):
        var = 1 / vector_l
        matr = np.random.normal(loc=0, scale=np.sqrt(var),
                                size=(vector_l, number_vec))
        matr_normed = matr / np.linalg.norm(matr, axis=0)
        return matr_normed

    def get_max_diff(vector):

        var = 1/len(vector[:, 0])

        def cum_gauß(x):
            return (1.0 + erf(x / np.sqrt(2.0*var))) / 2.0

        length = len(vector[:, 0])
        vector = np.sort(vector, axis=0).T
        dist = np.abs(cum_gauß(vector) -
                      np.array([i+1 for i in range(length)])/length)
        return np.max(dist, axis=1)

    vec = get_N_normalised_vec(vector_l=tested_vec, number_vec=number_test_s)
    max_differences = get_max_diff(vec)

    sorted_diff = np.sort(np.array(max_differences))
    y_axis = get_cdf(len(sorted_diff))

    def test_stat(x):
        return np.interp(x, sorted_diff, y_axis)
    results = 1 - test_stat(get_max_diff(evh.T))
    avg_indices = range(averaging_window, np.size(results) - averaging_window)
    p_values_avg = np.zeros(np.shape(avg_indices))
    for i in avg_indices:
        p_values_avg[i - averaging_window] = np.mean(
            results[i-averaging_window:i+averaging_window])
    return p_values_avg
