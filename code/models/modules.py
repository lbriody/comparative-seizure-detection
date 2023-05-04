import tensorflow as tf
    
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    
    def __init__(self, filters, *args, **kwargs):
      super().__init__(
        kernel_size=(3, 3),
        strides=1,
        **kwargs
      )
      self.filters = filters

    def build(self, input_shape):
      super().build(input_shape)
      self.pointwise_conv2d = tf.keras.layers.Conv2D(
        filters=self.filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=False
      )
    
    def call(self, inputs):
      x = super().call(inputs)
      return self.pointwise_conv2d(x)
    
    def get_config(self):
      config = super().get_config()
      config.update({
        'filters': self.filters
      })
      return config
    
    @classmethod
    def from_config(cls, config):
      return cls(**config)
    