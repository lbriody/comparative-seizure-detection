import tensorflow as tf
from wavetf import WaveTFFactory

class FourierModel(tf.keras.Model):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.conv1d_block = tf.keras.Sequential([
      tf.keras.layers.Conv1D(
        input_shape=(178, 1),
        filters=16,
        kernel_size=31,
        strides=1,
        padding='same',
        name='conv1d_1'
      ),
      tf.keras.layers.BatchNormalization(momentum=0.9),
      tf.keras.layers.ReLU(),
      tf.keras.layers.MaxPool1D(
        input_shape=(178, 16),
        pool_size=3,
        strides=2,
        name='maxpool1d_1'
      ),
      tf.keras.layers.DepthwiseConv1D(
        input_shape=(89, 16),
        depth_multiplier=16, 
        kernel_size=3, 
        strides=1,
        padding='same',
        name='depthwise_conv1d_1'
      ),
      tf.keras.layers.BatchNormalization(momentum=0.9),
      tf.keras.layers.ReLU(),
      tf.keras.layers.MaxPool1D(
        input_shape=(89, 16),
        pool_size=3,
        strides=2,
        name='maxpool1d_2'
      ),
      tf.keras.layers.DepthwiseConv1D(
        input_shape=(44, 16),
        depth_multiplier=16,
        kernel_size=3,
        strides=1,
        padding='same',
        name='depthwise_conv1d_2'
      ),
      tf.keras.layers.BatchNormalization(momentum=0.9),
      tf.keras.layers.ReLU(),
      tf.keras.layers.DepthwiseConv1D(
        input_shape=(44, 16),
        depth_multiplier=32,
        kernel_size=3,
        strides=1,
        padding='same',
        name='depthwise_conv1d_3'
      ),
      tf.keras.layers.BatchNormalization(momentum=0.9),
      tf.keras.layers.ReLU(),
      tf.keras.layers.MaxPool1D(
        input_shape=(44, 16),
        pool_size=3,
        strides=2,
        name='maxpool1d_3'
      ),
      tf.keras.layers.DepthwiseConv1D(
        input_shape=(22, 16),
        depth_multiplier=32,
        kernel_size=3,
        strides=1,
        padding='same',
        name='depthwise_conv1d_4'
      ),
      tf.keras.layers.BatchNormalization(momentum=0.9),
      tf.keras.layers.ReLU(),
      tf.keras.layers.GlobalAveragePooling1D()
    ])

    self.conv2d_block = tf.keras.Sequential([
      tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=(15,7),
        strides=1,
        padding='same',
        name='conv2d_1'
      ),
      tf.keras.layers.BatchNormalization(momentum=0.9),
      tf.keras.layers.ReLU(),
      tf.keras.layers.MaxPool2D(
        pool_size=(3,3),
        strides=2,
        name='max_pool_1',
        padding='same'
      ),
      tf.keras.layers.DepthwiseConv2D(
        depth_multiplier=16,
        kernel_size=(3,3),
        strides=1,
        padding='same',
        name='depthwise_conv2d_1'
      ),
      tf.keras.layers.BatchNormalization(momentum=0.9),
      tf.keras.layers.ReLU(),
      tf.keras.layers.MaxPool2D(
        pool_size=(3,3),
        strides=2,
        name='max_pool_2',
        padding='same'
      ),
      tf.keras.layers.DepthwiseConv2D(
        depth_multiplier=16,
        kernel_size=(3,3),
        strides=1,
        padding='same',
        name='depthwise_conv2d_2'
      ),
      tf.keras.layers.BatchNormalization(momentum=0.9),
      tf.keras.layers.ReLU(),
      tf.keras.layers.DepthwiseConv2D(
        depth_multiplier=32,
        kernel_size=(3,3),
        strides=1,
        padding='same',
        name='depthwise_conv2d_3'
      ),
      tf.keras.layers.BatchNormalization(momentum=0.9),
      tf.keras.layers.ReLU(),
      tf.keras.layers.MaxPool2D(
        pool_size=(3,3),
        strides=2,
        name='max_pool_3',
        padding='same'
      ),
      tf.keras.layers.DepthwiseConv2D(
        depth_multiplier=32,
        kernel_size=(3,3),
        strides=1,
        padding='same',
        name='depthwise_conv2d_4'
      ),
      tf.keras.layers.BatchNormalization(momentum=0.9),
      tf.keras.layers.ReLU(),
      tf.keras.layers.GlobalAveragePooling2D()
    ])

    self.dwt = WaveTFFactory().build(
      kernel_type='haar',
      dim=1
    )

    self.classifier = tf.keras.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128),
      tf.keras.layers.BatchNormalization(momentum=0.9),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Dense(
        units=1,
        activation='sigmoid',
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
      )
    ])

  @tf.function
  def call(self, inputs):
    eeg_conv = self.conv1d_block(inputs)
    stft_out = tf.abs(
      tf.signal.stft(
        inputs[..., 0], 
        frame_length=128, 
        frame_step=64,
        fft_length=128,
        pad_end=True,
        window_fn=tf.signal.hann_window,
      )
    )[..., tf.newaxis]
    stft_conv = self.conv2d_block(stft_out)
    fft_out = tf.signal.rfft(inputs)
    fft_conv = self.conv2d_block(fft_out)
    dwt_out = self.dwt(inputs)
    dwt_conv = self.conv1d_block(tf.reshape(dwt_out, [-1, inputs.shape[1], 1]))

    concat = tf.concat([eeg_conv, stft_conv, fft_conv, dwt_conv], axis=1)
    return self.classifier(concat)


  def get_config(self):
    super_config = super().get_config()
    return {
      'conv1d_block': self.conv1d_block,
      'conv2d_block': self.conv2d_block,
      'classifier': self.classifier,
      'dwt': self.dwt
    }
  
  @classmethod
  def from_config(cls, config):
    return cls(**config)
  