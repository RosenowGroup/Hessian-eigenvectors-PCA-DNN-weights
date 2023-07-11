(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train,x_test=x_train/255,x_test/255
# normalise the data

for i in range(x_train.shape[0]):
    x_train[i]-=np.mean(x_train[i],axis=(0,1))
    x_train[i]/=np.std(x_train[i],axis=(0,1))
for i in range(x_test.shape[0]):
    x_test[i]-=np.mean(x_test[i],axis=(0,1))
    x_test[i]/=np.std(x_test[i],axis=(0,1))

class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.c1=layers.Conv2D(300, (5, 5), activation='relu')
    self.p1=layers.MaxPooling2D((3, 3) , padding="same" )
    self.c2=layers.Conv2D(150, (3, 3), activation='relu')
    self.p2=layers.MaxPooling2D((3, 3) , padding="same" )
    self.flatten = layers.Flatten()
    self.d1 = layers.Dense(384, activation='relu',
          kernel_regularizer=tf.keras.regularizers.L2(1e-4))
    self.d2 = layers.Dense(192, activation='relu',
          kernel_regularizer=tf.keras.regularizers.L2(1e-4))
    self.d3 = layers.Dense(10, activation='softmax')

  def call(self, x):
    x=self.c1(x)
    x=self.p1(x)
    x=self.c2(x)
    x=self.p2(x)
    x = self.flatten(x)
    x = self.d1(x)
    x = self.d2(x)
    return self.d3(x)
model = MyModel()
batch_size=32
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
n_batches=train_ds.cardinality().numpy()
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay( # after 10 epochs 60% of initial learning rate
    0.01,
    decay_steps= y_train.size / batch_size,
    decay_rate=0.95,
    staircase=True)
epochs_train=100
final_learning_rate=0.01*0.95**epochs_train
model.build((32,32,32,3))