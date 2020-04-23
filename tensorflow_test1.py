import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()

#create data

x_data = np.random.rand(500).astype(np.float32)
y_data = x_data*0.1 + 0.3

###create tensorflow structure start ###

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

### create tensorflow structure end ###

sess = tf.Session()
sess.run(init)            ### very important

for step in range(1, 1001):
    sess.run(train)
    if step % 10 == 0:
        print(step, sess.run(Weights), sess.run(biases))
