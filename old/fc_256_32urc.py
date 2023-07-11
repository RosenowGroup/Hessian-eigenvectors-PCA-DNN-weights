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
    self.flatten = layers.Flatten()
    self.d1 = layers.Dense(256, activation='relu')
    self.d2 = layers.Dense(128, activation='relu')
    self.d3 = layers.Dense(64, activation='relu')
    self.d4 = layers.Dense(10, activation='softmax')

  def call(self, x):
    x = self.flatten(x)
    x = self.d1(x)
    x = self.d2(x)
    x = self.d3(x)
    return self.d4(x)
model = MyModel()
model.build((32,3072))

batch_size=32
epochs_train=200
lr_schedule = 1e-2
final_learning_rate=lr_schedule
