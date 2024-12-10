# Lib

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras. models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


#C:\Users\vaibh\AppData\Local\Temp\CUDA
print(tf.__version__)

with tf.device(':/GPU:0'):
    #Ensure Tensorflow uses GPU ONLY
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #print(f"Using GPU is CUDA:{os.environ['CUDA_VISIBLE_DEVICES']}")
    # from tensorflow.python.client import device_lib
    # print (device_lib.list_local_devices())

    #Load the dataset
    (X_train, y_train) , (X_test, y_test) = cifar10.load_data()

    #Normalization
    X_train = X_train.astype('float32')/255
    X_test = X_test.astype('float32')/255

    # Convert class labels to one-hot encoced vectors
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Define the labels of the dataset
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Print the shapes of the datasets to verify transformations
    print(f"X_train shape: {X_train.shape}")  # Should be (50000, 32, 32, 3)
    print(f"y_train shape after one-hot encoding: {y_train.shape}")  # Should be (50000, 10)
    print(f"X_test shape: {X_test.shape}")  # Should be (10000, 32, 32, 3)
    print(f"y_test shape after one-hot encoding: {y_test.shape}")  # Should be (10000, 10)

    # Define the output directory
    output_dir =  "output"
    #os.path.abspath(os.path.join(os.path.dirname(__file__), '../output'))

    # Define the plot directory within the output directory
    plot_path = os.path.join(output_dir, 'plots')

    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    def display_images(images, labels, y_data, rows=4, cols=4, save_path=None):
        fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
        axes = axes.ravel()
        for i in np.arange(0, rows * cols):
            index = np.random.randint(0, len(images))
            axes[i].imshow(images[index])
            label_index = np.argmax(y_data[index])  # Get the index of the label
            axes[i].set_title(labels[label_index])
            axes[i].axis('off')
        plt.subplots_adjust(hspace=0.5)
        if save_path:
            plt.savefig(save_path)
            print(f'Plot saved to {save_path}')
        plt.show()  # Show the plot
        plt.close()  # Close the figure after showing it

    # Define the file path to save the plot
    plot_file = os.path.join(plot_path, 'display_images.png')

    # Display a sample of training images with their labels and save the plot
    display_images(X_train, labels, y_train, save_path=plot_file)


    # define CNN Model
    def cnn_model():
        model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)), 
        MaxPooling2D((2,2)), 
        Conv2D(64,(3,3), activation='relu'), 
        MaxPooling2D((2,2)),
        Flatten(), 
        Dense(64, activation='relu'), 
        Dropout(0.5),
        Dense(10, activation='softmax')
        ])
        return model

    #Define the model path
    model_path = os.path.join(output_dir, 'cifar10_simple_model.h5')

    if os.path.isfile(model_path):
        model = tf.keras.models.load_model(model_path)
        print(f"loaded existing model from {model_path}")
    else:
        model = cnn_model()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
        model.save(model_path)

        # Plot the training and validation accuracy over epochs
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')

        # Save the plot to a file
        plot_file = os.path.join(plot_path, '02_02_end_simple_model.png')
        plt.savefig(plot_file)
        print(f'Plot saved to {plot_file}')

        plt.show()
    plt.show()  # Show the plot
    plt.close()  # Close the figure after showing it
    # Evaluate the model on the test data to get the loss and accuracy
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy}")
    
