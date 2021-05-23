import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix

EPOCHS = 20
BATCH_SIZE = 32
LOSS = keras.metrics.categorical_crossentropy
OPTIMIZER = keras.optimizers.SGD()
METRIC = ['accuracy']


def plot_errors(history, epochs):
    # Plotting Train and Validation Loss

    epochs_range = list(range(1, epochs + 1))
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(1, figsize=(10, 6))
    plt.plot(epochs_range, train_loss, label='train_loss')
    plt.plot(epochs_range, val_loss, label='validation_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Binary Cross Entropy')
    plt.legend()
    plt.show()

    return True


def plot_confusion_matrix(y_test, y_pred, classes=[]):
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)

    plt.figure(1, figsize=(16, 8))
    sns.set(font_scale=1.5, color_codes=True, palette='deep')
    sns.heatmap(cm_df, annot=True, annot_kws={'size': 16}, fmt='d', cmap='YlGnBu')
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title('Confusion Matrix')
    plt.show()


def model_training(model, X_train, y_train, X_cv, y_cv, epochs, batch_size,loss, optimizer, metrics):
    # Initialise optimizers
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # Enabling Early Stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)

    # Enabling check point
    mc = ModelCheckpoint(filepath='bestModel.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    # Model fitting
    history = model.fit(X_train, y_train, validation_data=(X_cv, y_cv), epochs=epochs, batch_size=batch_size,
                        verbose=1, callbacks=[es, mc])

    return model, history


class Lenet5:

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def model_init(self, kernel_size=(5, 5), pool_size=(2, 2)):
        if isinstance(kernel_size, tuple) and isinstance(pool_size, tuple):
            # Define the LeNet-5 architecture
            ip = Input(shape=self.input_shape)

            # 1st Convolutional Layer followed by Max pooling (6 kernels)
            x = Conv2D(filters=6, kernel_size=kernel_size, activation='tanh')(ip)
            x = MaxPooling2D(pool_size=pool_size)(x)

            # 2nd Convolution Layer followed by Max pooling (16 kernels)
            x = Conv2D(filters=16, kernel_size=kernel_size, activation='tanh')(x)
            x = MaxPooling2D(pool_size=pool_size)(x)

            # 3rd Convolution Layer followed by Flattening (120 Kernels)
            x = Conv2D(filters=120, kernel_size=kernel_size, activation='tanh')(x)
            x = Flatten()(x)

            # Fully Connected Dense Layer
            x = Dense(units=84, activation='tanh')(x)

            # Final Output Softmax(layer)
            op = Dense(units=self.output_shape, activation='softmax')(x)

            # Define Model
            model = Model(inputs=ip, outputs=op)

            model.summary()

            return model


if __name__ == '__main__':
    # Loading the dataset and perform splitting
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Resize image to 32x 32
    X_train = np.array([np.pad(X_train[i], pad_width=2) for i in range(X_train.shape[0])])
    X_test = np.array([np.pad(X_test[i], pad_width=2) for i in range(X_test.shape[0])])

    # Performing reshaping operation
    X_train = X_train.reshape(X_train.shape[0], 32, 32, 1)
    X_test = X_test.reshape(X_test.shape[0], 32, 32, 1)

    # Normalization
    X_train = X_train / 255
    X_test = X_test / 255

    # One Hot Encoding
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Define image size and number of classes
    image_size = X_train.shape[1:]
    classes = y_train.shape[1]

    # Create model instance and initialise Lenet Model
    lenet = Lenet5(image_size, classes)
    model = lenet.model_init()

    epochs = EPOCHS
    batch_size = BATCH_SIZE
    loss = LOSS
    optimizer = OPTIMIZER
    metric = METRIC
    model, history = model_training(model, X_train, y_train, X_test, y_test, epochs, batch_size,
                                    loss, optimizer, metric)

    # Plot trainning and test error
    plot_errors(history, epochs)

    # Perform Prediction
    y_pred = model.predict(X_test)

    # Get list of prediction
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    # Show Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, list(range(classes)))
