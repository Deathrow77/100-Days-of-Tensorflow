import tensorflow as tf
mat_a = tf.random_normal((10,10), mean=0, stddev=1.5)
mat_b = tf.random_uniform((10,10), minval=-4, maxval=4)
mat_c = mat_a + mat_b
sess = tf.InteractiveSession()
print(mat_c.eval())
