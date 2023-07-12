import numpy as np
import tensorflow as tf


def train(
        model,
        epochs_train,
        lr_schedule,
        x_train,
        y_train,
        x_test,
        y_test,
        batch_size,
        layer_dtype=np.float32):
    """
    Trains a tensorflow model.

    Args:
      model: compiled model.
      epochs_train: number of epochs to train.
      lr_schedule: learning rate or schedule to train the network with
      x_train: training samples.
      y_train: training labels.
      x_test: test samples.
      y_test: test labels.
      batch_size: batch size.

    Returns:
      Trained model and history.
    """
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).batch(batch_size)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='test_accuracy')

    # train function
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_fn(labels, predictions)
            loss += sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    # test function
    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = loss_fn(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)
    history = np.zeros((epochs_train, 4), dtype=layer_dtype)
    for epoch in range(epochs_train):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)
        history[epoch] = [train_loss.result(), train_accuracy.result(),
                          test_loss.result(), test_accuracy.result()]
        print(
            f'Epoch: {epoch}'
            f'Loss: {train_loss.result():.2e}, '
            f'Accuracy: {train_accuracy.result():.6e}, '
            f'Test Loss: {test_loss.result():.2e}, '
            f'Test Accuracy: {test_accuracy.result():.2e}'
        )
    return model, history


def comp_train(
        model,
        epochs_train,
        x_train,
        y_train,
        x_test,
        y_test,
        batch_size):
    """
    Trains a compiled tensorflow model.

    Args:
      model: compiled model.
      epochs_train: number of epochs to train.
      x_train: training samples.
      y_train: training labels.
      x_test: test samples.
      y_test: test labels.
      batch_size: batch size.

    Returns:
      Trained model and history.
    """
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs_train,
                        verbose=2,
                        validation_data=(x_test, y_test))
    return model, history.history


def weight_measure(
        model,
        layer_str,
        epochs_train,
        lr_schedule,
        x_train,
        y_train,
        batch_size,
        measure_epochs_only=False):
    """
    Trains a model and measure its weights.

    Args:
      model: model.
      layer_str: layer str where to compute the weights
      epochs_train: number of epochs to train.
      x_train: training samples.
      y_train: training labels.
      batch_size: batch size.
      measure_epochs_only: wether the weights are measured
        only once per epoch oder on every batch defaults to False.

    Returns:
      Trained model and weights.
    """
    if layer_str[:6] == "layers":
        layer_name = getattr(model, "layers")[int(
            layer_str[7:(len(layer_str)-1)])]
    else:
        layer_name = getattr(model, layer_str)
    layer_size = np.prod(layer_name.kernel.shape)
    layer_dtype = np.float32

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(batch_size)

    n_batches = train_ds.cardinality().numpy()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    if measure_epochs_only:
        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = loss_object(labels, predictions)
                loss += sum(model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)
        weights = np.zeros((epochs_train, layer_size), dtype=layer_dtype)
        for epoch in range(epochs_train):
            train_loss.reset_states()
            print('epoch: '+str(epoch)+' of '+str(epochs_train))

            for images, labels in train_ds:
                train_step(images, labels)
            templ = layer_name.get_weights()[0].flatten()
            weights[epoch] = templ
            print(train_loss.result())
    else:
        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = loss_object(labels, predictions)
                loss += sum(model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)
        weights = np.zeros((epochs_train*n_batches, layer_size), dtype=layer_dtype)
        index_1 = 0
        for epoch in range(epochs_train):
            train_loss.reset_states()
            print('epoch: '+str(epoch)+' of '+str(epochs_train))

            for images, labels in train_ds:
                train_step(images, labels)
                templ = layer_name.get_weights()[0].flatten()
                weights[index_1] = templ
                index_1 += 1
            print(train_loss.result())
    return model, weights
