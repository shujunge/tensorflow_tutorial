import time
import tensorflow as tf
tf.enable_eager_execution()
tf.executing_eagerly()

tfe=tf.contrib.eager

def measure(x, steps):
  # TensorFlow initializes a GPU the first time it's used, exclude from timing.
  tf.matmul(x, x)
  start = time.time()
  for i in range(steps):
    x = tf.matmul(x, x)
    _ = x.numpy()  # Make sure to execute op and not just enqueue it
  end = time.time()
  return end - start

shape = (2000, 2000)
steps = 200
print("Time to multiply a {} matrix by itself {} times:".format(shape, steps))

# Run on CPU:
with tf.device("/cpu:0"):
  print("CPU: {} secs".format(measure(tf.random_normal(shape), steps)))

# Run on GPU, if available:
if tfe.num_gpus() > 0:
  with tf.device("/gpu:0"):
    print("GPU0: {} secs".format(measure(tf.random_normal(shape), steps)))
  with tf.device("/gpu:1"):
    print("GPU1: {} secs".format(measure(tf.random_normal(shape), steps)))
  with tf.device("/gpu:2"):
    print("GPU2: {} secs".format(measure(tf.random_normal(shape), steps)))
else:
  print("GPU: not found")


