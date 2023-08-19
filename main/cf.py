import numpy as np
import tensorflow as tf
import hess
import analysis


def get_layer_range(model):
    """
    Extract an array of layer indices.

    Args:
        model: model.
    Returns:
        Array of layer indices.
    """
    index = 0
    layer_range = []
    for layer in model.layers:
        if len(layer.get_weights()) > 0:
            layer_range.append(index)
        index += 1
    return np.array(layer_range)


def get_tasks(images, labels, task):
    """
    Extract tasks from a dataset.

    Args:
        images: images to compute the Hessian.
        labels: labels to compute the Hessian.
        task: List or array of labels that should be in the task.
    Returns:
        Images, labels of the task.
    """
    boolies = np.isin(labels[:, 0], task)
    return np.compress(boolies, images, axis=0), np.extract(boolies, labels)


def get_hess(model, layer_pointer, number_cons, images, labels, batch_size=1000):
    """
    Computes the Hessian eigenvectors of a layer tensor times its eigenvalue.

    Args:
        model: model.
        layer_pointer: pointer to the tensor.
        images: images to compute the Hessian.
        labels: labels to compute the Hessian.
        number_cons: number of parameter to conserve.
    Returns:
        Hessian eigenvectors in shape (number_cons,layer_pointer.shape).
    """
    if np.prod(layer_pointer.shape) >= number_cons*2:
        ew, ev = hess.lancz_single(
            model, layer_pointer, number_cons, images, labels, batch_size)
    else:
        ew, ev = hess.hess_lp(model, layer_pointer,
                              number_cons, images, labels, batch_size)
        ew = ew[:number_cons]
        ev = ev[:number_cons]
    return ((ew*ev.T).T).reshape((number_cons, *layer_pointer.shape))


def get_sv(layer_pointer, number_cons):
    """
    Computes the singular vectors of a layer matrix.

    Args:
        layer_pointer: pointer to matrix.
        number_cons: number of parameter to conserve, list for kernel and bias,
            can be in in parts or number.

    Returns:
        Singular vector matrix in shape (number_cons,layer_pointer.shape).
    """
    if len(layer_pointer.shape) > 2:
        matrix = analysis.svd_conv(layer_pointer)
        svec = analysis.svd(matrix)[:number_cons]
        return svec.reshape((number_cons, *layer_pointer.shape))
    else:
        matrix = layer_pointer
        svec = analysis.svd(matrix)[:number_cons]
        return svec.reshape((number_cons, *layer_pointer.shape))


def get_normed_regs(regs):
    """
    Norms the regularizations, such that the largest vector has a norm of one for each layer.

    Args:
        regs: List of regularizations.

    Returns:
        List of normed regularziations.
    """
    normed_regs = []
    for reg in regs:
        normed_regs.append(reg/np.linalg.norm(reg[0]))
    return normed_regs


def create_reg(model, images, labels, number_cons=[2, 2],
               hess_only=False, batch_size=1000):
    """
    Computes regularizers for layers.

    Args:
        model: model where the reg should be added.
        images: images to compute the Hessian.
        labels: labels to compute the Hessian.
        number_cons: number of parameter to conserve, list for kernel and bias,
            can be in in parts or number.
        hess_only: optional, whether the singular value method should be used,
            or only the Hessian decomposition,
            defaults to False using the sv method.
        batch_size: optional, batch size of training samples, must be set small enough
          such that the GPU does not run out of memory.

    Returns:
        List of regularizers in the ordering of model.layers.
    """
    regs = []
    for layer in model.trainable_variables:
        if len(layer.shape) > 2:
            if hess_only:
                if number_cons[0] >= 1:
                    number_sv = number_cons[0]
                else:
                    number_sv = int(
                        np.prod(layer.shape)*number_cons[0])+1
                temp = get_hess(model, layer,
                                number_sv, images, labels, batch_size)
            else:
                if number_cons[0] >= 1:
                    if len(layer.shape) > 2:
                        if number_cons[0] > int(np.min([np.prod(layer.shape[:2]), np.prod(
                                layer.shape[::-1][:2])])):
                            number_sv = int(np.min([np.prod(layer.shape[:2]), np.prod(
                                layer.shape[::-1][:2])]))-1
                        else:
                            number_sv = number_cons[0]
                    else:
                        if number_cons[0] > int(np.min(
                                [layer.shape[0], layer.shape[1]])):
                            number_sv = int(np.min(
                                [layer.shape[0], layer.shape[1]]))-1

                        else:
                            number_sv = number_cons[0]
                else:
                    if len(layer.shape) > 2:
                        number_sv = int(np.min([np.prod(layer.shape[:2]), np.prod(
                            layer.shape[::-1][:2])])*number_cons[0])+1
                    else:
                        number_sv = int(np.min(
                            [layer.shape[0], layer.shape[1]])*number_cons[0])+1
                temp = get_sv(layer, number_sv)
            regs.append(tf.convert_to_tensor(temp))

        else:
            if number_cons[1] >= 1:
                number_b = number_cons[1]
            else:
                number_b = int((layer.shape)
                               [0]*number_cons[1])+1
            temp = get_hess(model, layer,
                            number_b, images, labels, batch_size)
            regs.append(temp)
    return regs


def cf_loss(model, lbd, regs, weights_1):
    """
    Computes the loss of the catastrophic forgetting regularization.

    Args:
        lbd: regularization constant.
        regs: List of regularization tensors.
        model: model.
        weights_1: weights of the former task.
    Returns:
        List of losses.
    """
    losses = []
    for vec, weights, weights_conserve in zip(regs,
                                              model.trainable_weights,
                                              weights_1):
        red_vec = tf.scalar_mul(lbd, tf.math.reduce_sum(
            tf.square(tf.multiply(
                tf.subtract(weights, weights_conserve), vec))))
        losses.append(red_vec)
    return losses


def train_cf(model, lbd, regs, weights_1,
             epochs_train, lr_schedule,
             x_train, y_train, x_train_1,
             y_train_1, x_test_1, y_test_1, x_test_2, y_test_2,
             batch_size, layer_dtype=np.float32):
    """
    Trains the network for the second task with
      catastrophic forgetting regularizer.

    Args:
        model: model.
        lbd: regularization constant.
        regs: List of regularization tensors.
        weights_1: weights of the former task.
        epochs_train: number of epochs to train.
        lr_schedule: learning rate or schedule to train the network with.
        x_train: training samples.
        y_train: training labels.
        x_train_1: training samples of first task.
        y_train_1: training labels of first task.
        x_test_1: test samples of first task.
        y_test_1: test labels of first task.
        x_test_2: test samples of second task.
        y_test_2: test labels of second task.
        batch_size: batch size.
    Returns:
        Trained model, history.
    """
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(batch_size)
    test_ds_2 = tf.data.Dataset.from_tensor_slices(
        (x_test_2, y_test_2)).batch(batch_size)
    train_ds_1 = tf.data.Dataset.from_tensor_slices(
        (x_train_1, y_train_1)).batch(batch_size)
    test_ds_1 = tf.data.Dataset.from_tensor_slices(
        (x_test_1, y_test_1)).batch(batch_size)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='test_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape(persistent=True) as tape:
            predictions = model(images, training=True)
            loss = loss_fn(labels, predictions)
            loss += sum(model.losses)
            loss += sum(cf_tf_loss(model, lbd, regs, weights_1))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = loss_fn(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)
    cf_tf_loss = tf.function(cf_loss)
    history = np.zeros((epochs_train, 4), dtype=layer_dtype)
    for epoch in range(epochs_train):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        for images, labels in train_ds:
            train_step(images, labels)
        for test_images, test_labels in test_ds_2:
            test_step(test_images, test_labels)
        test_2 = test_accuracy.result()
        test_loss.reset_states()
        test_accuracy.reset_states()
        for test_images, test_labels in train_ds_1:
            test_step(test_images, test_labels)
        train_1 = test_accuracy.result()
        test_loss.reset_states()
        test_accuracy.reset_states()
        for test_images, test_labels in test_ds_1:
            test_step(test_images, test_labels)
        test_1 = test_accuracy.result()
        history[epoch] = [train_accuracy.result(), test_2, train_1, test_1]
        print(
            f'Epoch: {epoch}'
            f'Loss: {train_loss.result():.2e}, '
            f'Train Accuracy: {train_accuracy.result():.6e}, '
            f'Test Error 1: {test_1:.2e}, '
            f'Test Error 2: {test_2:.2e}'
        )
    return model, history


def cf_gradient(gamma, gradients, regs):
    """
    Computes the gradient of the catastrophic forgetting regularization.

    Args:
        gamma: regularization factor.
        gradients: list of gradients
          of which the regularization should be applied on.
        regs: list of regularization tensors.
    Returns:
        List of gradients.
    """
    new_gradients = []
    for grad, vec in zip(gradients, regs):
        reduceDims = tf.range(1, tf.rank(grad)+1)
        red_vec = gamma * \
            tf.tensordot(tf.math.reduce_sum(tf.multiply(
                grad, vec), axis=reduceDims), vec, axes=[0, 0])
        new_gradients.append(grad-red_vec)
    return new_gradients


def train_cf_grad(model, gamma, regs, epochs_train, lr_schedule,
                  x_train, y_train,
                  x_train_1, y_train_1,
                  x_test_1, y_test_1,
                  x_test_2, y_test_2,
                  batch_size, layer_dtype=np.float32):
    """
    Trains the network for the second task with
      catastrophic forgetting regularizer on the gradient.

    Args:
        model: model.
        gamma: regularization factor.
        regs: list of regularization tensors.
        epochs_train: number of epochs to train.
        lr_schedule: learning rate or schedule to train the network with.
        x_train: training samples.
        y_train: training labels.
        x_train_1: training samples of first task.
        y_train_1: training labels of first task.
        x_test_1: test samples of first task.
        y_test_1: test labels of first task.
        x_test_2: test samples of second task.
        y_test_2: test labels of second task.
        batch_size: batch size.
    Returns:
        Trained model, history.
    """
    normed_regs = get_normed_regs(regs)
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(batch_size)
    test_ds_2 = tf.data.Dataset.from_tensor_slices(
        (x_test_2, y_test_2)).batch(batch_size)
    train_ds_1 = tf.data.Dataset.from_tensor_slices(
        (x_train_1, y_train_1)).batch(batch_size)
    test_ds_1 = tf.data.Dataset.from_tensor_slices(
        (x_test_1, y_test_1)).batch(batch_size)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='test_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape(persistent=True) as tape:
            predictions = model(images, training=True)
            loss = loss_fn(labels, predictions)
            # loss_reg=sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        cf_gradients = cf_tf_gradient(gamma, gradients, normed_regs)
        """reg_grads=tape.gradient(loss_reg,model.trainable_variables)
      tot_grads=[]
      for reg_grad,grad in zip(reg_grads,cf_gradients):
        tot_grads.append(reg_grad+grad)"""
        optimizer.apply_gradients(zip(cf_gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = loss_fn(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)
    cf_tf_gradient = tf.function(cf_gradient)
    history = np.zeros((epochs_train, 4), dtype=layer_dtype)
    for epoch in range(epochs_train):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds_2:
            test_step(test_images, test_labels)
        test_2 = test_accuracy.result()
        test_loss.reset_states()
        test_accuracy.reset_states()
        for test_images, test_labels in train_ds_1:
            test_step(test_images, test_labels)
        train_1 = test_accuracy.result()
        test_loss.reset_states()
        test_accuracy.reset_states()
        for test_images, test_labels in test_ds_1:
            test_step(test_images, test_labels)
        test_1 = test_accuracy.result()
        history[epoch] = [train_accuracy.result(), test_2, train_1, test_1]
        print(
            f'Epoch: {epoch}'
            f'Loss: {train_loss.result():.2e}, '
            f'Train Accuracy: {train_accuracy.result():.6e}, '
            f'Test Error 1: {test_1:.2e}, '
            f'Test Error 2: {test_2:.2e}'
        )
    return model, history
