import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Creating the dataset

x_train = np.linspace(0, 10, 10) + np.random.normal(-1, 1, 10)
y_train = np.linspace(0, 10, 10) + np.random.normal(-1, 1, 10)

# Creating the model

# y = mx + b where m and b need to be calculated

m = tf.Variable(0.47)
b = tf.Variable(1.00)

error = 0

for x, y in zip(x_train, y_train):
    # Calculate the predicted value as per the new values of m and b
    y_hat = m*x + b
    # Update the error ( here mean squared ) 
    error += (y - y_hat)**2

# Run an optimizer to minimize the value of error

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

# Execute the preceding part and train the model
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    epochs = 100
    curr_epoch = 0
    for i in range(epochs):
        sess.run(train)
        curr_epoch = curr_epoch + 1
        print("Current Epoch - " + str(curr_epoch) + " Error - " + str(error.eval()))
    f_m, f_c = sess.run([m, b]) 

# Generating the test set
x_test = np.linspace(-10,15,10)
y_pred = f_m*x_test + f_c

# Plotting the value
plt.plot(x_train, y_train, 'r*')
plt.plot(x_test, y_pred, 'g--')
plt.show()



