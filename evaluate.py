import tensorflow as tf
import config
from prepare_data import get_datasets
import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report

train_generator, valid_generator, test_generator, \
train_num, valid_num, test_num= get_datasets()


# Load the model
model = tf.keras.models.load_model(config.model_dir)
#model = tf.keras.models.load_model('best_model_Adam_lr3_noAu.h5')
#model = model.load_weights('best_weight_lr3.h5')

# Perform predictions on test data

print("..... Testing in progress.....")
y_pred = model.predict_generator(test_generator)
y_pred = np.array([np.argmax(x) for x in y_pred])


y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred)
print(cm)

print(classification_report(y_true, y_pred))
