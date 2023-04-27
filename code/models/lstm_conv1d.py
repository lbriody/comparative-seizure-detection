# model 2: lstm with conv1d

import tensorflow as tf
from tensorflow import keras

class LSTMConvModel(keras.Model):
  
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  @tf.function
  def call(self, inputs):
    pass

  def get_config(self):
    super_config = super().get_config()
    return {
      **super_config
    }
  
  @classmethod
  def from_config(cls, config):
    return cls(**config)