# Inspired by Krzysztof Sopyla - krzysztofsopyla@gmail.com - https://github.com/ksopyla/svm_mnist_digit_classification
# License: MIT

import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


# Print digits along with a label explaining predicted or labeled value
def printDigitExamples(images, targets, imageTitle, sample_size=24, title_text='Digit {}'):
    nsamples = sample_size
    rand_idx = np.random.choice(images.shape[0], nsamples)
    images_and_labels = list(zip(images[rand_idx], targets[rand_idx]))
    img = plt.figure(1, figsize=(15, 12), dpi=160)
    img.suptitle(imageTitle)
    for index, (image, label) in enumerate(images_and_labels):
        plt.subplot(np.ceil(nsamples / 6.0), 6, index + 1)
        plt.axis('off')
        plt.imshow(image.reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(title_text.format(label))
    plt.show()


# Download MNIST dataset
print("Fetch dataset at " + str(datetime.datetime.now()))
mnist = fetch_openml('mnist_784')
print("Dataset downloaded at " + str(datetime.datetime.now()))

# Load Dataset
images = mnist.data
targets = mnist.target
print("Dataset loaded at " + str(datetime.datetime.now()))

# Print examples from the training set
print("Printing traning example at " + str(datetime.datetime.now()))
printDigitExamples(images, targets, "Training Set Example")

# Standardize dataset imformation for SciKit-Learn
X_data = images / 255.0
Y = targets

# Split dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y, test_size=0.15, random_state=42)

# Create classifier object
model = svm.SVC(C=5, gamma=0.05)

# Mark start time
start = datetime.datetime.now()
print('Start learning at {}'.format(str(start)))

# Train model
model.fit(X_train, Y_train)

# Calculate training time
elapsed_time = datetime.datetime.now() - start
print("Model trained")
print("Time to train model: " + (str(elapsed_time)))

# Test model with remaining data
modeledData = model.predict(X_test)

# Print examples of predicted test data
printDigitExamples(X_test, modeledData, "Prediction Example", title_text="Predicted {}")

# Generate confusion matrix
cm = metrics.confusion_matrix(Y_test, modeledData)

# Plot accuracy for each number in XKCD graph
numberAccuracy = list()
numbers = list()
for i in range(0, 10):
    numbers.append(i)
    numberAccuracy.append(round((cm[i][i] / sum(cm[i])) * 100, 2))

plt.xkcd(scale=1.15, randomness=5)
fig = plt.figure()
ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
ax.bar(numbers, numberAccuracy, 0.7)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks(numbers)
ax.set_xticklabels(numbers)
ax.set_xlim([-0.5, 10])
ax.set_yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
ax.set_yticklabels([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
ax.set_ylim([0, 110])

ax.set_title("Prediction Accuracy in % per Number")

fig.text(
    0.5, 0.05,
    'Total accuracy = ' + str(round(metrics.accuracy_score(Y_test, modeledData) * 100, 2)) + "%",
    ha='center')

plt.show()
