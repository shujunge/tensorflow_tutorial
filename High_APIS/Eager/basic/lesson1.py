from __future__ import absolute_import, division, print_function
import tensorflow as tf
##########eager setting###########
tf.enable_eager_execution()
tf.executing_eagerly()
##################################

x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))

a = tf.constant([[1, 2],
                 [3, 4]])
b=tf.add(1,a)
m=tf.multiply(a,b)
print(m.numpy)


def fizzbuzz(max_number):
  counter=tf.constant(0)
  for num in range(max_number):
    num=tf.constant(num)
    if int(num%3)==0 and int(num%5)==0:
      counter+=1
  return counter

counter=fizzbuzz(100)
print(counter)

###编写自己的层
class mysmaple(tf.keras.layers.Layer):
  def __init__(self,output_unints):
    self.output_units=output_unints
  def build(self,input):
    self.kernel=self.add_variable("kernel",
                                  [input.shape[-1],self.output_units])
  def call(self,input):
    return tf.matmul(input,self.kernel)

##Eager Model
model=tf.keras.Sequential([
  tf.keras.layers.Dense(10,input_shape=(10,)),
tf.keras.layers.Dense(1)])
model.summary()
######################

##Eager training
tfe = tf.contrib.eager
w = tfe.Variable([[1.0]])
with tf.GradientTape() as tape:
  loss = w * w
grad = tape.gradient(loss, [w])
print(grad)  # => [tf.Tensor([[ 2.]], shape=(1, 1), dtype=float32)]



####################################demo1#########################################
NUM_EXAMPLES = 1000
training_inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

def prediction(input, weight, bias):
  return input * weight + bias

# A loss function using mean-squared error
def loss(weights, biases):
  error = prediction(training_inputs, weights, biases) - training_outputs
  return tf.reduce_mean(tf.square(error))

# Return the derivative of loss with respect to weight and bias
def grad(weights, biases):
  with tf.GradientTape() as tape:
    loss_value = loss(weights, biases)
  return tape.gradient(loss_value, [weights, biases])

train_steps = 200
learning_rate = 0.01
# Start with arbitrary values for W and B on the same batch of data
W = tfe.Variable(5.)
B = tfe.Variable(10.)

print("Initial loss: {:.3f}".format(loss(W, B)))

for i in range(train_steps):
  dW, dB = grad(W, B)
  W.assign_sub(dW * learning_rate)
  B.assign_sub(dB * learning_rate)
  if i % 20 == 0:
    print("Loss at step {:03d}: {:.3f}".format(i, loss(W, B)))

print("Final loss: {:.3f}".format(loss(W, B)))
print("W = {}, B = {}".format(W.numpy(), B.numpy()))


##############################demon2 mnist###########################################


tfe = tf.contrib.eager
class Model(tf.keras.Model):
  def __init__(self):
    super(Model, self).__init__()
    self.W = tfe.Variable(5., name='weight')
    self.B = tfe.Variable(10., name='bias')
  def predict(self, inputs):
    return inputs * self.W + self.B

# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 2000
training_inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

# The loss function to be optimized
def loss(model, inputs, targets):
  error = model.predict(inputs) - targets
  return tf.reduce_mean(tf.square(error))

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, model.variables)

# Define:
# 1. A model.
# 2. Derivatives of a loss function with respect to model parameters.
# 3. A strategy for updating the variables based on the derivatives.
model = Model()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

print("Initial loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))

# Training loop
for i in range(300):
  grads = grad(model, training_inputs, training_outputs)
  optimizer.apply_gradients(zip(grads, model.variables),
                            global_step=tf.train.get_or_create_global_step())
  if i % 20 == 0:
    print("Loss at step {:03d}: {:.3f}".format(i, loss(model, training_inputs, training_outputs)))

print("Final loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))

##模型的保存
import os
model = Model()
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
checkpoint_dir ='/weights'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
root = tfe.Checkpoint(optimizer=optimizer,
                      model=model,
                      optimizer_step=tf.train.get_or_create_global_step())
root.save(file_prefix=checkpoint_prefix)
root.restore(tf.train.latest_checkpoint(checkpoint_dir))

##Summaries and TensorBoard
logdir='summary'
writer = tf.contrib.summary.create_file_writer(logdir)
global_step=tf.train.get_or_create_global_step()  # return global step var
writer.set_as_default()
iterations=1000
for _ in range(iterations):
  global_step.assign_add(1)
  # Must include a record_summaries method
  with tf.contrib.summary.record_summaries_every_n_global_steps(100):
    # your model code goes here
    tf.contrib.summary.scalar('loss', loss)
    
## Use eager execution in a graph environment
def my_py_func(x):
  x = tf.matmul(x, x)  # You can use tf ops
  print(x)  # but it's eager!
  return x

with tf.Session() as sess:
  x = tf.placeholder(dtype=tf.float32)
  # Call eager function in graph!
  pf = tfe.py_func(my_py_func, [x], tf.float32)
  sess.run(pf, feed_dict={x: [[2.0]]})  # [[4.0]]
