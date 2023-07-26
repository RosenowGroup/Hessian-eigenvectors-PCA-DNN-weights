import tensorflow as tf
import numpy as np
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train/255, x_test/255
# normalise the data
for i in range(x_train.shape[0]):
    x_train[i] -= np.mean(x_train[i], axis=(0, 1))
    x_train[i] /= np.std(x_train[i], axis=(0, 1))
for i in range(x_test.shape[0]):
    x_test[i] -= np.mean(x_test[i], axis=(0, 1))
    x_test[i] /= np.std(x_test[i], axis=(0, 1))


def lrs_str_to_learning_rate(lrs_str, init_lr, y_train_size, batch_size):
    if lrs_str[:4] == "NONE":
        learning_rate = init_lr
    elif lrs_str[:2] == "ED":
        decay_rate = float(lrs_str[2:6])
        num_batches = int(np.ceil(y_train_size/batch_size))
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=init_lr,
            decay_steps=num_batches,
            decay_rate=decay_rate,
            staircase=True)
    elif lrs_str[:4] == "PW01":
        num_batches = int(np.ceil(y_train_size/batch_size))
        boundaries = [80*num_batches, 120*num_batches]
        values = [init_lr, 0.1*init_lr, 0.01*init_lr]
        learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries, values)
    else:
        raise ValueError("unkown learning rate schedule")
    return learning_rate


def cfr10_lenet_create(
    dtype="float32",
    weight_decay=1e-4,
    init_lr=5e-3,
    momentum=0.9,
    batch_size=50,
    lrs_str="ED0.98x100"
):
    """
    Args:
      dtype: string containing the datatype of the network, float32 or float64.
      weight_decay: parameter for L2 regularization.
      init_lr: initial learning rate, might change if lr schedule is used.
      momentum: parameter for heavy ball momentum.
      batch_size: example batch size for update gradient.
      lrs_str: containing information about the learning rate schedule, if no
        schedule is used, the string equals NONExabc, with 
        abc = number_of_epochs_trained.

    Returns:
      A LeNet for Cifar-10 as a compiled keras model with the desired parameters.
    """

    y_train_size = 50000
    tf.keras.backend.set_floatx(dtype)

    # learning rate schedule initialization
    learning_rate = lrs_str_to_learning_rate(
        lrs_str, init_lr, y_train_size, batch_size)

    # model initialization
    initializer = tf.keras.initializers.GlorotNormal()
    l2_reg = tf.keras.regularizers.L2(weight_decay)

    # model architecture
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(6, 5, activation='relu', input_shape=(
        32, 32, 3), padding='same', kernel_initializer=initializer, kernel_regularizer=l2_reg))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(16, 5, activation='relu', padding='same',
              kernel_initializer=initializer, kernel_regularizer=l2_reg))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(120, activation='relu',
              kernel_initializer=initializer, kernel_regularizer=l2_reg))
    model.add(tf.keras.layers.Dense(84, activation='relu',
              kernel_initializer=initializer, kernel_regularizer=l2_reg))
    model.add(tf.keras.layers.Dense(10, activation='softmax',
              kernel_initializer=initializer, kernel_regularizer=l2_reg))

    # compile model
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    return model


batch_size = 64
model = cfr10_lenet_create(batch_size=batch_size)

model.build((batch_size, 3072))

init_lr = 5e-3
lrs_str = "ED0.98x100"
epochs_train = int(lrs_str[-3:])
lr_schedule = lrs_str_to_learning_rate(
    lrs_str, init_lr, y_train.shape[0], batch_size)
final_learning_rate = init_lr*float(lrs_str[2:6])**epochs_train
