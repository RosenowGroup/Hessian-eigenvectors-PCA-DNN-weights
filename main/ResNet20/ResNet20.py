(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train,x_test=x_train/255,x_test/255
# normalise the data


for i in range(x_train.shape[0]):
    x_train[i]-=np.mean(x_train[i],axis=(0,1))
    x_train[i]/=np.std(x_train[i],axis=(0,1))
for i in range(x_test.shape[0]):
    x_test[i]-=np.mean(x_test[i],axis=(0,1))
    x_test[i]/=np.std(x_test[i],axis=(0,1))
def lrs_str_to_learning_rate(lrs_str, init_lr, y_train_size, batch_size):
  if lrs_str[:4] == "NONE":
    learning_rate = init_lr
  elif lrs_str[:2] == "ED":
    decay_rate = float(lrs_str[2:6])
    num_batches = int(np.ceil(y_train_size/batch_size))
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate = init_lr, 
      decay_steps= num_batches,
      decay_rate=decay_rate,
      staircase=True)
  elif lrs_str[:4] == "PW01":
    num_batches = int(np.ceil(y_train_size/batch_size))
    boundaries = [80*num_batches, 120*num_batches]
    values = [init_lr, 0.1*init_lr, 0.01*init_lr]
    learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
  else: 
    raise ValueError("unkown learning rate schedule")
  return learning_rate

def cfr10_ResNet20BN_create(
    dtype = "float32", 
    seed = 1, 
    weight_decay = 1e-4, 
    init_lr = 1e-3, 
    momentum = 0.9, 
    batch_size = 128, 
    lrs_str = "PW01x160"
    ):
  """
  Args:
    dtype: string containing the datatype of the network, float32 or float64.
    seed: seed for weight initialization.
    weight_decay: parameter for L2 regularization.
    init_lr: initial learning rate, might change if lr schedule is used.
    momentum: parameter for heavy ball momentum.
    batch_size: example batch size for update gradient.
    lrs_str: containing information about the learning rate schedule, if no
      schedule is used, the string equals NONExabc, with 
      abc = number_of_epochs_trained.

  Returns:
    ResNet20 for Cifar-10 with batch-normalization as a compiled keras model 
    with the desired parameters.
  """

  # The model name helps to identify saved data later on
  model_type = "CFR10_ResNet20BN"
  model_name = (model_type+"_"+dtype[-2:]+"_"+f"{seed:.0f}_wd{weight_decay:.0e}_lr{init_lr:.0e}_b{batch_size:.0f}_"+lrs_str)
  y_train_size = 50000
  num_res_blocks = 3
  tf.keras.backend.set_floatx(dtype)

  # learning rate schedule initialization
  learning_rate = lrs_str_to_learning_rate(lrs_str, init_lr, y_train_size, batch_size)

  # model initialization
  tf.random.set_seed(seed)
  initializer = tf.keras.initializers.GlorotNormal(seed=seed)
  l2_reg = tf.keras.regularizers.L2(weight_decay) 

  # model architecture
  def residual_blockBN(x, initializer, regularizer, filters=16, num_res_blocks=3, downsampeling=False):
    if downsampeling == True:
      y = tf.keras.layers.Conv2D(filters, 3, strides=(2,2), padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer)(x)
      x = tf.keras.layers.Conv2D(filters, 1, padding='same', strides=(2,2), kernel_initializer=initializer, kernel_regularizer=regularizer)(x)
    else:
      y = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer)(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation('relu')(y)
    y = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    x = tf.keras.layers.Add()([x,y])
    x = tf.keras.layers.Activation('relu')(x)
    for idx in range(1, num_res_blocks):
      y = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer)(x)
      y = tf.keras.layers.BatchNormalization()(y)
      y = tf.keras.layers.Activation('relu')(y)
      y = tf.keras.layers.Conv2D(filters, 3, padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer)(y)
      y = tf.keras.layers.BatchNormalization()(y)
      x = tf.keras.layers.Add()([x,y])
      x = tf.keras.layers.Activation('relu')(x)
    return x

  inputs = tf.keras.layers.Input(shape=(32, 32, 3))
  x = tf.keras.layers.Conv2D(16, 3, padding='same', kernel_initializer=initializer, kernel_regularizer=l2_reg)(inputs)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = residual_blockBN(x, initializer=initializer, regularizer=l2_reg, filters=16, num_res_blocks=num_res_blocks, downsampeling=False)
  x = residual_blockBN(x, initializer=initializer, regularizer=l2_reg, filters=32, num_res_blocks=num_res_blocks, downsampeling=True)
  x = residual_blockBN(x, initializer=initializer, regularizer=l2_reg, filters=64, num_res_blocks=num_res_blocks, downsampeling=True)
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Flatten()(x)
  outputs = tf.keras.layers.Dense(10, activation='softmax', kernel_initializer=initializer, kernel_regularizer=l2_reg)(x)

  model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)

  # compile model
  model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate = learning_rate, momentum = momentum), 
                loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
  
  return model
model = cfr10_ResNet20BN_create()
batch_size=128
model.build((batch_size,3072))
epochs_train=160
lr_schedule = lrs_str_to_learning_rate("PW01x160", 1e-3, y_train.shape[0], batch_size)
#final_learning_rate=1e-4
