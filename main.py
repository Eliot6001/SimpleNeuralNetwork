import numpy as np
from matplotlib import pyplot as plt
import zipfile
import os 

zip_file_path = './archive (1).zip'
force_train = False 

# Data Shuffle
with zipfile.ZipFile(zip_file_path, 'r') as z:
    with z.open('train-images.idx3-ubyte', 'r') as f:
        f.read(16)
        train_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)

with zipfile.ZipFile(zip_file_path, 'r') as z:
    with z.open('train-labels.idx1-ubyte', 'r') as f:
        f.read(8)
        train_labels = np.frombuffer(f.read(), dtype=np.uint8)

with zipfile.ZipFile(zip_file_path, 'r') as z:
    with z.open('t10k-images.idx3-ubyte', 'r') as f:
        f.read(16)
        test_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)

with zipfile.ZipFile(zip_file_path, 'r') as z:
    with z.open('t10k-labels.idx1-ubyte', 'r') as f:
        f.read(8)
        test_labels = np.frombuffer(f.read(), dtype=np.uint8)

# Combined
combined_images = np.concatenate((train_images, test_images), axis=0)
combined_labels = np.concatenate((train_labels, test_labels), axis=0)

shuffled_indices = np.random.permutation(len(combined_images))
split_point = 55000

train_indices = shuffled_indices[:split_point]
test_indices = shuffled_indices[split_point:]

new_train_images = combined_images[train_indices]
new_train_labels = combined_labels[train_indices]

new_test_images = combined_images[test_indices]
new_test_labels = combined_labels[test_indices]

# Reshape and normalize
training_images = new_train_images.reshape(55000, 784).T / 255.0
validation_images = new_test_images.reshape(15000, 784).T / 255.0

print(f"Training images shape: {test_labels.shape}")
print(f"Training labels: {validation_images.shape}")

# Initialize parameters with a standard deviation of 0.1 for better results
def init_params():
    W1 = np.random.randn(10, 784) * 0.01
    b1 = np.zeros((10, 1))
    W2 = np.random.randn(10, 10) * 0.01
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2

def rectify(x):
    return np.maximum(x, 0)

# Softmax function, now with axis=0 to handle multi-sample batches correctly
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def forwardprop(w1, b1, w2, b2, x):
    X1 = w1.dot(x) + b1
    a1 = rectify(X1)
    X2 = w2.dot(a1) + b2  # FIX: Changed b1 to b2
    a2 = softmax(X2)
    return X1, a1, X2, a2

def reLU(z): #backpropagation stuff
    return np.greater(z, 0).astype(int)

def onehot(Y):
    y_hot = np.zeros((Y.size, Y.max() + 1))
    y_hot[np.arange(Y.size), Y] = 1
    return y_hot.T

def backwardprop(z1, a1, z2, a2, w1, w2, x, y):
    y_hot = onehot(y)
    m = y.size
    
    dz2 = a2 - y_hot
    dw2 = (1 / m) * dz2.dot(a1.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
    
    dz1 = w2.T.dot(dz2) * reLU(z1)
    dw1 = (1 / m) * dz1.dot(x.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
    
    return dw1, db1, dw2, db2


def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2


def get_predictions(a2):
    return np.argmax(a2, 0)

def get_accuracy(predictions, y):
    return np.sum(predictions == y) / y.size

def gradient_descent(x, y, alpha, iterations):
    if not force_train and all(os.path.exists(f"{file}.npy") for file in ['w1', 'b1', 'w2', 'b2']):
        print("Found existing model files. Loading parameters and skipping training.")
        w1 = np.load('w1.npy')
        b1 = np.load('b1.npy')
        w2 = np.load('w2.npy')
        b2 = np.load('b2.npy')
        return w1, b1, w2, b2
    w1, b1, w2, b2 = init_params()

    for i in range(iterations):
        z1, a1, z2, a2 = forwardprop(w1, b1, w2, b2, x)
        
        dw1, db1, dw2, db2 = backwardprop(z1, a1, z2, a2, w1, w2, x, y)
        
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        
        if i % 10 == 0:
            print(f"Iteration: {i}")
            predictions = get_predictions(a2)
            print(f"Accuracy: {get_accuracy(predictions, y)}")
            
    np.save('w1.npy', w1)
    np.save('b1.npy', b1)
    np.save('w2.npy', w2)
    np.save('b2.npy', b2)
    
    return w1, b1, w2, b2



alpha = 0.1 # learning rate
iterations = 1500

w1, b1, w2, b2 = gradient_descent(training_images, new_train_labels, alpha, iterations)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forwardprop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = training_images[:, index, None]
    prediction = make_predictions(training_images[:, index, None], W1, b1, W2, b2)
    label = new_train_labels[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.draw()

def make_single_prediction(image, w1, b1, w2, b2):
    reshaped_image = image.reshape(784, 1) / 255.0
    _, _, _, a2 = forwardprop(w1, b1, w2, b2, reshaped_image)
    prediction = np.argmax(a2, 0)
    return prediction



x = make_predictions(validation_images, w1, b1, w2, b2)
accuracy = get_accuracy(x, new_test_labels)

print(f"Validation Accuracy: {accuracy}")

