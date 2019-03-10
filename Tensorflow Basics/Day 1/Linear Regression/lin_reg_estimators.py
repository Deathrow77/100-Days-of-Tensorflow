import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Generating our dataset 
np.random.seed(101)
x_data = np.random.normal(0.0,100.0,200)
y_data = np.random.normal(0.0,100.0,200)

# Defining Features Column and giving a key to the feature set

feature_col = [tf.feature_column.numeric_column('x', shape=(1))]

# Defining an estimator

estimator = tf.estimator.LinearRegressor(feature_columns=feature_col)

# Train Test split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

# Defining the input functions 

input_funct = tf.estimator.inputs.numpy_input_fn({'x':x_train}, y_train, shuffle=True, batch_size=4, num_epochs=None)

train_input_funct = tf.estimator.inputs.numpy_input_fn({'x':x_train}, y_train, batch_size=4, shuffle=False, num_epochs=1000)

test_input_funct = tf.estimator.inputs.numpy_input_fn({'x':x_test}, y_test, batch_size=4, shuffle=False, num_epochs=1000)

# Training the estimator

estimator.train(input_fn=input_funct, steps=1000)

# Evaluating the metrics

train_metrics = estimator.evaluate(input_fn=train_input_funct, steps=1000)

test_metrics = estimator.evaluate(input_fn=test_input_funct, steps=1000)
print("\n\n\n")
print("Train Metrics -- \n")
print(train_metrics)
print("\n\n\n")
print("Test Metrics -- \n")
print(test_metrics)
print("\n\n\n")
# Making predictions on new dataset
new_data = np.random.normal(-100.0,300, 200)

# Input Function for predicting values in the new dataset
predict_input_funct = tf.estimator.inputs.numpy_input_fn({'x':new_data}, shuffle=False)

# Extracting the predicted values from the predictions dictionary into a list
predictions = []
for prediction in list(estimator.predict(input_fn=predict_input_funct)):
    predictions.append(prediction['predictions'])

# Visualizing the predictions
plt.plot(x_data, y_data, 'r*')
plt.plot(new_data, predictions, 'g--')
plt.show()