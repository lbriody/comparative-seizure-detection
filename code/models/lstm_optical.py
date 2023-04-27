import tensorflow as tf


class LSTMOpticalModel(tf.keras.Model):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # TODO: Implement architecture

  @tf.function
  def call(self, inputs):
    # TODO: Implement forward pass
    pass

  def get_config(self):
    super_config = super().get_config()
    return {
      **super_config
    }
  
  @classmethod
  def from_config(cls, config):
    return cls(**config)
