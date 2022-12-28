import tensorflow as tf
import config

def get_datasets():
    # Preprocess the dataset
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=40,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip = True,
        fill_mode='nearest',
        validation_split=0.2)


    train_generator = train_datagen.flow_from_directory(config.train_dir,
                                                        subset='training',
                                                        target_size=(config.image_height, config.image_width),
                                                        color_mode="rgb",
                                                        batch_size=config.BATCH_SIZE,
                                                        seed=123,
                                                        shuffle=True,
                                                        class_mode='binary'
                                                        )

    valid_generator = train_datagen.flow_from_directory(config.train_dir,
                                                        subset='validation',
                                                        target_size=(config.image_height, config.image_width),
                                                        color_mode="rgb",
                                                        batch_size=config.BATCH_SIZE,
                                                        seed=123,
                                                        shuffle=True,
                                                        class_mode='binary')


    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 /255.0
    )
    test_generator = test_datagen.flow_from_directory(config.test_dir,
                                                      target_size=(config.image_height, config.image_width),
                                                      color_mode="rgb",
                                                      batch_size=config.BATCH_SIZE,
                                                      seed=123,
                                                      shuffle=False,
                                                      class_mode='binary'
                                                      )


    train_num = train_generator.samples
    valid_num = valid_generator.samples
    test_num = test_generator.samples


    return train_generator, \
           valid_generator, \
           test_generator, \
           train_num, valid_num, test_num
