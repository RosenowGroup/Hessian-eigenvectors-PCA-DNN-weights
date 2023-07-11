(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train,x_test=x_train/255,x_test/255
# normalise the data

for i in range(x_train.shape[0]):
    x_train[i]-=np.mean(x_train[i],axis=(0,1))
    x_train[i]/=np.std(x_train[i],axis=(0,1))
for i in range(x_test.shape[0]):
    x_test[i]-=np.mean(x_test[i],axis=(0,1))
    x_test[i]/=np.std(x_test[i],axis=(0,1))
reg=tf.keras.regularizers.l2(5e-4)
class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.flatten = layers.Flatten()
    self.d1 = layers.Dense(256, activation='relu',kernel_regularizer=reg)
    self.d2 = layers.Dense(128, activation='relu',kernel_regularizer=reg)
    self.d3 = layers.Dense(64, activation='relu',kernel_regularizer=reg)
    self.d4 = layers.Dense(10, activation='softmax',kernel_regularizer=reg)

  def call(self, x):
    x = self.flatten(x)
    x = self.d1(x)
    x = self.d2(x)
    x = self.d3(x)
    return self.d4(x)
model = MyModel()
model.build((32,3072))

batch_size=32
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
n_batches=train_ds.cardinality().numpy()
epochs_train=200
initial_learning_rate = 1e-2
decay_rate=0.99
final_learning_rate=initial_learning_rate*decay_rate**epochs_train
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=n_batches,
    decay_rate=decay_rate,
    staircase=True)
