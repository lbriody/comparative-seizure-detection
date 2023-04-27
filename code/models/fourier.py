import tensorflow as tf
class FourierModel(tf.keras.Model):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.conv1d_block = tf.keras.Sequential([
      tf.keras.layers.Conv1D(
        filters=16,
        kernel_size=31,
        strides=1
      ),
      tf.keras.layers.BatchNormalization(momentum=0.9),
      tf.keras.layers.ReLU(),
      tf.keras.layers.MaxPool1D(
        pool_size=3,
        strides=2
      ),
      tf.keras.layers.DepthwiseConv1D(
        filters=16,
        kernel_size=3,
        strides=1
      ),
      tf.keras.layers.BatchNormalization(momentum=0.9),
      tf.keras.layers.ReLU(),
      tf.keras.layers.MaxPool1D(
        pool_size=3,
        strides=2
      ),
      tf.keras.layers.DepthwiseConv1D(
        filters=16,
        kernel_size=3,
        strides=1
      ),
      tf.keras.layers.BatchNormalization(momentum=0.9),
      tf.keras.layers.ReLU(),
      tf.keras.layers.DepthwiseConv1D(
        filters=32,
        kernel_size=3,
        strides=1
      ),
      tf.keras.layers.BatchNormalization(momentum=0.9),
      tf.keras.layers.ReLU(),
      tf.keras.layers.MaxPool1D(
          pool_size=3,
          strides=2
      ),
      tf.keras.layers.DepthwiseConv1D(
        filters=32,
        kernel_size=3,
        strides=1
      ),
      tf.keras.layers.BatchNormalization(momentum=0.9),
      tf.keras.layers.ReLU(),
      tf.keras.layers.GlobalAveragePooling1D()
    ])

    self.conv2d_block = tf.keras.Sequential([
      tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=(15,7),
        strides=1
      ),
      tf.keras.layers.BatchNormalization(momentum=0.9),
      tf.keras.layers.ReLU(),
      tf.keras.layers.MaxPool2D(
        pool_size=(3,3),
        strides=2,
      ),
      tf.keras.layers.DepthwiseConv2D(
        filters=16,
        kernel_size=(3,3),
        strides=1
      ),
      tf.keras.layers.BatchNormalization(momentum=0.9),
      tf.keras.layers.ReLU(),
      tf.keras.layers.MaxPool2D(
        pool_size=(3,3),
        strides=2
      ),
      tf.keras.layers.DepthwiseConv2D(
        filters=16,
        kernel_size=(3,3),
        strides=1
      ),
      tf.keras.layers.BatchNormalization(momentum=0.9),
      tf.keras.layers.ReLU(),
      tf.keras.layers.DepthwiseConv2D(
        filters=32,
        kernel_size=(3,3),
        strides=1
      ),
      tf.keras.layers.BatchNormalization(momentum=0.9),
      tf.keras.layers.ReLU(),
      tf.keras.layers.MaxPool2D(
        pool_size=(3,3),
        strides=2
      ),
      tf.keras.layers.DepthwiseConv2D(
        filters=32,
        kernel_size=(3,3),
        strides=1
      ),
      tf.keras.layers.BatchNormalization(momentum=0.9),
      tf.keras.layers.ReLU(),
      tf.keras.layers.GlobalAveragePooling2D()
    ])

    self.classifier = tf.keras.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(
        units=2,
        activation='softmax',
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
      )
    ])

  @tf.function
  def call(self, inputs):
    eeg_conv = self.conv1d_block(inputs)
    stft_out = tf.signal.stft(inputs, frame_length=128, frame_step=512, fft_length=128)
    stft_conv = self.conv2d_block(stft_out)
    fft_out = tf.signal.fft(tf.cast(inputs, dtype=tf.complex64)))
    fft_conv = self.conv2d_block(fft_out)
    dwt_out = tf.signal.dwt(inputs, wavelet='db1')
    dwt_conv = self.conv1d_block(dwt_out)

    concat = tf.concat([eeg_conv, stft_conv, fft_conv, dwt_conv], axis=1)
    return self.classifier(concat)


  def get_config(self):
    super_config = super().get_config()
    return {
      **super_config
    }
  
  @classmethod
  def from_config(cls, config):
    return cls(**config)
  