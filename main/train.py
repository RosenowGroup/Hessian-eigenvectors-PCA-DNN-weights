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
      Trained model and histroy.
    """
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs_train,
                        verbose=2,
                        validation_data=(x_test, y_test))
    return model, history.history
