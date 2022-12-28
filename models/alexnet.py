import tensorflow as tf
image_height = 224
image_width = 224
channels = 3


def AlexNet():
    model = tf.keras.Sequential([
        # layer 1
        tf.keras.layers.Conv2D(filters=96,
                               kernel_size=(11, 11),
                               strides=4,
                               padding="valid",
                               activation=tf.keras.activations.relu,
                               input_shape=(image_height, image_width, channels)),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=2,
                                  padding="valid"),
        tf.keras.layers.BatchNormalization(),
        # layer 2
        tf.keras.layers.Conv2D(filters=256,
                               kernel_size=(5, 5),
                               strides=1,
                               padding="same",
                               activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=2,
                                  padding="same"),
        tf.keras.layers.BatchNormalization(),
        # layer 3
        tf.keras.layers.Conv2D(filters=384,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same",
                               activation=tf.keras.activations.relu),
        # layer 4
        tf.keras.layers.Conv2D(filters=384,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same",
                               activation=tf.keras.activations.relu),
        # layer 5
        tf.keras.layers.Conv2D(filters=256,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same",
                               activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=2,
                                  padding="same"),
        tf.keras.layers.BatchNormalization(),
        # layer 6
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=4096,
                              activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(rate=0.2),
        # layer 7
        tf.keras.layers.Dense(units=4096,
                              activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(rate=0.2),
        # layer 8
        tf.keras.layers.Dense(units=1,
                              activation=tf.keras.activations.sigmoid)
    ])

    return model