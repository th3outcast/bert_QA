import tensorflow as tf
import numpy as np
import os
from tflite_model_maker import model_spec
from tflite_model_maker import question_answer
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.question_answer import DataLoader

spec = model_spec.get('mobilebert_qa_squad')

train_data_path = tf.keras.utils.get_file(
    fname='triviaqa-web-train-8000.json',
    origin='https://storage.googleapis.com/download.tensorflow.org/models/tflite/dataset/triviaqa-web-train-8000.json')
validation_data_path = tf.keras.utils.get_file(
    fname='triviaqa-verified-web-dev.json',
    origin='https://storage.googleapis.com/download.tensorflow.org/models/tflite/dataset/triviaqa-verified-web-dev.json')

train_data = DataLoader.from_squad(train_data_path, spec, is_training=True)
validation_data = DataLoader.from_squad(validation_data_path, spec, is_training=False)

model = question_answer.create(train_data, model_spec=spec)

model.summary()

model.evaluate(validation_data)

model.export(export_dir='.')
