import numpy as np
import tensorflow as tf
from tensorflow import keras

EPOCHS = 50  # was 75
BATCH_SIZE = 100  # best - 90 # was 50


def create_dataset(xs, ys, n_classes):
    ys = tf.one_hot(ys, depth=n_classes)
    return tf.data.Dataset.from_tensor_slices((xs, ys)) \
        .shuffle(ys.shape[0]) \
        .batch(BATCH_SIZE)


def train(train_data, test_data):
    # Setup training data (No pre-process data manipulation done --> Normalize?)
    x_train = train_data
    y_train = x_train.pop("ACTION")

    # Reserve 10,000 samples for validation
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]

    # Create your datasets
    train_dataset = create_dataset(x_train, y_train, 2)
    val_dataset = create_dataset(x_val, y_val, 2)

    # 3-Layer model:
    # model = keras.Sequential([  # consider an Attention layer
    #     keras.layers.Dense(units=256, activation='relu'),
    #     keras.layers.Dense(units=2, activation='softmax')  # 2 neurons b/c 2 access classes
    # ])

    # Deep neural network: 5 Layer
    model = keras.Sequential([  # consider an Attention layer
        keras.layers.Dense(units=512, activation='selu'),
        keras.layers.Dense(units=256, activation='elu'),
        keras.layers.Dense(units=8, activation='relu'),
        keras.layers.Dense(units=2, activation='softmax')  # 2 neurons b/c 2 access classes
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_dataset.repeat(),
        epochs=EPOCHS,
        steps_per_epoch=1000,  # best - 100 # was 5000
        validation_data=val_dataset.repeat(),
        validation_steps=2
    )

    # Run prediction on test data
    x_test = test_data.drop(["id"], axis=1)
    predictions = model.predict(x_test)
    y_test_predict = np.argmax(predictions, axis=1)

    return y_test_predict

