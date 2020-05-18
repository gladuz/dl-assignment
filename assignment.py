import numpy as np
#Importing helper functions from tools.py. They are moved to separate file in order to increase readability
from tools import accuracy, check_accuracy, ReLU, dReLU, cost, mse

def generate_data():
    first_class = np.random.multivariate_normal((1, 1), [[1,0], [0,1]], 50)
    second_class = np.random.multivariate_normal((-1, -1), [[1,0], [0,1]], 50)
    data = np.array(list(zip(np.concatenate((first_class, second_class)), [[1, 0]]*50 + [[0, 1]]*50)))
    np.random.shuffle(data)
    return data[:, 0], data[:, 1]

training_data, training_label = generate_data()
test_data, test_label = generate_data()

#Batch size and learning rate constants
batch_size = 50
learning_rate = 0.1

#Initialize weights
w0 = np.random.normal(0, 0.001, (2, 10))
w1 = np.random.normal(0, 0.001, (10, 10))
w2 = np.random.normal(0, 0.001, (10, 2))

#Forward pass
def forward(X, w0, w1, w2):
    #Initialize biases
    b0 = np.zeros((1, 10))
    b1 = np.zeros((1, 10))
    b2 = np.zeros((1, 2))

    #First hidden layer
    z1 = X.dot(w0) + b0
    a1 = ReLU(z1)

    #Second hidden layer
    z2 = a1.dot(w1) + b1
    a2 = ReLU(z2)

    #Output layer
    z3 = a2.dot(w2) + b2
    return z1, z2, z3

#Check the accuracy of the model before back propagation
check_accuracy(test_data, test_label, w0, w1, w2, forward)
print("\nStarting the gradient descent...\n")

#Gradient descent
for i in range(100):
    #Select the random batch from the dataset
    random_indices = np.random.choice(training_data.shape[0], batch_size, replace=False)
    X = training_data[random_indices]
    y = training_label[random_indices]

    #Forward propagation
    z1, z2, z3 = forward(X, w0, w1, w2)

    #Backpropagation
    d3 = cost(z3, y)
    d2 = np.dot(w2, d3.T).T * dReLU(z2)
    d1 = np.dot(w1, d2.T).T * dReLU(z1)
    
    #Calculate gradient update for weights
    dw2 = np.dot(z2.T, d3)
    dw1 = np.dot(z1.T, d2)
    dw0 = np.dot(X.T, d1)

    #Updating weights
    w0 = w0 - learning_rate*dw0
    w1 = w1 - learning_rate*dw1
    w2 = w2 - learning_rate*dw2

    #Check the accuracy of the model and print it every ten steps
    if(i % 10 == 9):
        check_accuracy(test_data, test_label, w0, w1, w2, forward, step=i)
