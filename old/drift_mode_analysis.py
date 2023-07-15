# import dependencies
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
import time

model_str='fc_256_32r'
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
trained_epochs=0

# choose Parameter
learning_rate = 1e-3
batch_size = 32
epochs=1100
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train,x_test=x_train/255,x_test/255
# normalise the data

for i in range(x_train.shape[0]):
    x_train[i]-=np.mean(x_train[i],axis=(0,1))
    x_train[i]/=np.std(x_train[i],axis=(0,1))
for i in range(x_test.shape[0]):
    x_test[i]-=np.mean(x_test[i],axis=(0,1))
    x_test[i]/=np.std(x_test[i],axis=(0,1))
# set the seed
tf.random.set_seed(1)


train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(batch_size)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
#train function
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss=loss_fn(labels, predictions)
    loss+=sum(model.losses)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)
#test function
@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_fn(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)
epochs
history = np.zeros((epochs, 4))
time_stemp = time.monotonic()
time_stemp_0 = time_stemp
tr_loss=1
epoch=0
while(tr_loss>1e-3):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  history[epoch] = np.array([train_loss.result(), train_accuracy.result(), test_loss.result(), test_accuracy.result()])
  time_temp = time.monotonic()
  time_diff = time_temp - time_stemp
  time_stemp = time_temp
  time_total = time_temp - time_stemp_0
  tr_loss=train_loss.result()
  epoch=+1
  print(
    f'{time_diff:.0f}s, '
    f'Loss: {train_loss.result():.2e}, '
    f'Accuracy: {train_accuracy.result():.2e}, '
    f'Test Loss: {test_loss.result():.2e}, '
    f'Test Accuracy: {test_accuracy.result():.2e}'
    )
  if epoch==epochs-1:
    break
layer_name=model.d3
#computes the number of parameters of the layer
layer_size=1
for i in layer_name.get_weights()[0].shape:
    layer_size*=i
#conserve the numpy datatype
layer_dtype=layer_name.get_weights()[0].dtype
layer_shape=layer_name.get_weights()[0].shape
n_batches=train_ds.cardinality().numpy()
dyn_epochs = 10 #min int(layer_size/n_batches)

weights=np.zeros((dyn_epochs*n_batches,layer_size),dtype=layer_dtype)
index_1=0
for epoch in range(dyn_epochs):
  train_loss.reset_states()
  for images, labels in train_ds:
    train_step(images, labels)
    templ=layer_name.get_weights()[0].flatten()
    weights[index_1]=templ
    index_1+=1
  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result():.2e}'
    )  
#calculate the covariance matrix
#np.save('weights/dynamic_weights_'+model_str+'_'+str(learning_rate),weights)

import numpy as np
from sklearn.decomposition import PCA

pca = PCA(n_components=4,svd_solver="randomized")
pca.fit(weights)
theta=np.tensordot(pca.components_,weights,axes=(1,1))
ind=np.argsort(np.var(theta,axis=1))[::-1]
pcomp=np.take(pca.components_, ind, axis=0)
theta=np.tensordot(pcomp,weights,axes=(1,1))
var=np.var(theta,axis=1)
drift_ratio=var[0]/var[1]

w_0, b = layer_name.get_weights()
# calculate the projection of pc to the weights
theta=np.tensordot(pcomp,weights,axes=(1,1))
theta_last=np.tensordot(pcomp,w_0.flatten(),axes=(1,0))
# adding the principal components together up to ith indices
def sum_weights(i):
    templ=np.zeros(pca.components_.shape[1])
    for j in range(i):
        templ+=theta_last[j]*pca.components_[j]
    return templ
@tf.function
def calc_acc(images, labels):
    test_accuracy.reset_states()
    model.reset_states()
    predictions = model(images, training=False)
    test_acc=test_accuracy(labels, predictions)
    tf.print(test_acc)
    return test_acc

layer_name.set_weights([sum_weights(1).reshape(layer_shape),b])
test_acc=calc_acc(x_test, y_test)

np.save('drift/drift_mode_'+str(learning_rate),theta[0])
np.save('drift/drift_ratio_'+str(learning_rate),np.array([learning_rate, drift_ratio, test_acc]))
