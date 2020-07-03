import json
import numpy as np
import tensorflow as tf 
import tensorflow.keras as keras 
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers

DATA_PATH = "data.json"
SAVED_MODEL = "model.h5"

LEARNING_RATE = 0.0001
EPOCHS = 40
BATCH_SIZE = 32
NUM_USED_DATA_CLASS = 6

def load_dataset(data_path):
    
    with open(data_path, "r") as fp:
        data = json.load(fp)
        
    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    
    return X, y
        


def get_data_splits(data_path, test_size=0.2, validation_size=0.2):
    
    
    X, y = load_dataset(data_path)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    
    
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape, learning_rate=0.0001, loss="sparse_categorical_crossentropy"):
    
    
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=input_shape, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))
  
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))
 
    model.add(tf.keras.layers.Conv2D(32, (2, 2), activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))
   
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
  
    model.add(tf.keras.layers.Dense(NUM_USED_DATA_CLASS, activation="softmax")) # [0.1, 0.7, 0.2]
  
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
 
    model.summary()
    
    return model
    

def main():
    
    
    X_train, X_validation, X_test, y_train, y_validation, y_test = get_data_splits(DATA_PATH)
    
   
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) #(* Segments, * Coefficient 13, *1)
    model = build_model(input_shape, learning_rate=LEARNING_RATE)
    
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_validation, y_validation))
 
    test_error, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test error: {test_error}, test accuracy: {test_accuracy}")
 
    model.save(SAVED_MODEL)
    
if __name__ == "__main__":
    main()