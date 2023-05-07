import tensorflow as tf
from wavetf import WaveTFFactory

class FourierModel(tf.keras.Model):

  def __init__(self, num_feats, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.weight_initializer = tf.keras.initializers.GlorotNormal(seed=69)
    self.to_channels_first = tf.keras.layers.Reshape((1, -1))
    self.to_channels_last = tf.keras.layers.Reshape((-1, 1))
    self.conv1d_block = tf.keras.Sequential([
      self.to_channels_first,
      tf.keras.layers.Conv1D(
        filters=16,
        kernel_size=31,
        strides=1,
        padding='same',
        name='conv1d_1',
        kernel_initializer=self.weight_initializer,
      ),
      self.to_channels_last,
      tf.keras.layers.BatchNormalization(momentum=0.9),
      tf.keras.layers.ReLU(),
      tf.keras.layers.MaxPool1D(
        pool_size=3,
        strides=2,
        name='maxpool1d_1'
      ),
      self.to_channels_first,
      tf.keras.layers.DepthwiseConv1D(
        depth_multiplier=16, 
        kernel_size=3, 
        strides=1,
        padding='same',
        name='depthwise_conv1d_1',
        depthwise_initializer=self.weight_initializer,
      ),
      self.to_channels_last,
      tf.keras.layers.BatchNormalization(momentum=0.9),
      tf.keras.layers.ReLU(),
      tf.keras.layers.MaxPool1D(
        pool_size=3,
        strides=2,
        name='maxpool1d_2'
      ),
      self.to_channels_first,
      tf.keras.layers.DepthwiseConv1D(
        depth_multiplier=16,
        kernel_size=3,
        strides=1,
        padding='same',
        name='depthwise_conv1d_2',
        depthwise_initializer=self.weight_initializer,
      ),
      tf.keras.layers.BatchNormalization(momentum=0.9),
      tf.keras.layers.ReLU(),
      tf.keras.layers.DepthwiseConv1D(
        depth_multiplier=32,
        kernel_size=3,
        strides=1,
        padding='same',
        name='depthwise_conv1d_3',
        depthwise_initializer=self.weight_initializer,
      ),
      self.to_channels_last,
      tf.keras.layers.BatchNormalization(momentum=0.9),
      tf.keras.layers.ReLU(),
      tf.keras.layers.MaxPool1D(
        pool_size=3,
        strides=2,
        name='maxpool1d_3'
      ),
      self.to_channels_first,
      tf.keras.layers.DepthwiseConv1D(
        depth_multiplier=32,
        kernel_size=3,
        strides=1,
        padding='same',
        name='depthwise_conv1d_4',
        depthwise_initializer=self.weight_initializer,
      ),
      self.to_channels_last,
      tf.keras.layers.BatchNormalization(momentum=0.9),
      tf.keras.layers.ReLU(),
      tf.keras.layers.GlobalAveragePooling1D(data_format='channels_first'),
    ], name='conv1d_block')

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
      tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')
    ], name='conv2d_block')

    self.dwt = WaveTFFactory().build(
      kernel_type='haar',
      dim=1,
    )

    self.classifier = tf.keras.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(
        num_feats * 32, 
        kernel_initializer=self.weight_initializer,
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
      ),
      tf.keras.layers.BatchNormalization(momentum=0.9),
      tf.keras.layers.ReLU(),
      tf.keras.layers.Dense(
        units=2,
        activation='softmax',
        kernel_initializer=self.weight_initializer,
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
      )
    ], name='classifier')

  @tf.function
  def call(self, inputs):
    print("inputs: ", inputs.shape)
    eeg_conv = self.conv1d_block(inputs)
    print("eeg_conv: ", eeg_conv.shape)
    stft_out = tf.cast(
      x=tf.signal.stft(
        inputs[..., 0],
        frame_length=128, 
        frame_step=8, 
        fft_length=128,
        pad_end=True,
        window_fn=tf.signal.hamming_window
      ),
      dtype=tf.float32
    )[:, 1:-1, :, tf.newaxis]
    print("stft_out: ", stft_out.shape)
    stft_conv = self.conv2d_block(stft_out)
    print("stft_conv: ", stft_conv.shape)
    fft_out = tf.cast(tf.signal.fft(tf.cast(inputs, tf.complex64)), tf.float32)
    fft_conv = self.conv1d_block(fft_out)
    print("fft_conv: ", fft_conv.shape)
    dwt_out = self.dwt(inputs)
    print(dwt_out.shape)
    dwt_conv = self.conv1d_block(tf.reshape(dwt_out, [-1, 178, 1]))
    print("dwt_conv: ", dwt_conv.shape)

    # concat = tf.concat([eeg_conv, stft_conv, fft_conv, dwt_conv], axis=1)
    concat = tf.concat([eeg_conv, fft_conv, dwt_conv], axis=1)
    print("concat: ", concat.shape)
    return self.classifier(concat)

  def get_config(self):
    super_config = super().get_config()
    super_config.update({
      'conv1d_block': self.conv1d_block,
      'conv2d_block': self.conv2d_block,
      'classifier': self.classifier,
      'dwt': self.dwt
    })
    return super_config
  
  @classmethod
  def from_config(cls, config):
    return cls(**config)
  