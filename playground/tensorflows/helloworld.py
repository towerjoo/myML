import tensorflow as tf

train_x = [1,2,3,4]
train_y = [0, -1, -2, -3]

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()


#session
sess = tf.Session()
sess.run(init)

N = 1000
for i in range(N):
    sess.run(train, {x: train_x, y: train_y})

print sess.run([W, b])


