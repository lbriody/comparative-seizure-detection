import tensorflow as tf
from wavetf import WaveTFFactory

class FourierModel(tf.keras.Model):

  def __init__(self, 
               eeg=False,
               fft=False,
               dwt=False,
               stft=False,
               one_d=False,
               full=False,
               *args, 
               **kwargs):
    super().__init__(*args, **kwargs)
    self.weight_initializer = tf.keras.initializers.GlorotNormal(seed=69)
    self.to_channels_first = tf.keras.layers.Reshape((1, -1))
    self.to_channels_last = tf.keras.layers.Reshape((-1, 1))
    
    self.eeg = eeg or full or one_d
    self.fft = fft or full or one_d
    self.dwt = dwt or full or one_d
    self.stft = stft or full

    if not(self.eeg or self.fft or self.dwt or self.stft):
      raise ValueError('At least one representation of EEG data must be specified to initialize FourierModel.')
    
    self.num_feats = 0
    if self.eeg: self.num_feats += 1
    if self.fft: self.num_feats += 1
    if self.dwt: self.num_feats += 1
    if self.stft: self.num_feats += 1
    
    if self.eeg or self.fft or self.dwt: 
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
    else: self.conv1d_block = None

    if self.stft:
      self.conv2d_block = tf.keras.Sequential([
        tf.keras.layers.Reshape((1,21,65)),
        tf.keras.layers.Conv2D(
          filters=16,
          kernel_size=(15,7),
          strides=1,
          padding='same',
          name='conv2d_1',
          kernel_initializer=self.weight_initializer
        ),
        tf.keras.layers.Reshape((21,16,1)),
        tf.keras.layers.BatchNormalization(momentum=0.9),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(
          pool_size=(3,3),
          strides=2,
          name='max_pool_1',
          padding='same'
        ),
        tf.keras.layers.Reshape((1,8,11)),
        tf.keras.layers.DepthwiseConv2D(
          depth_multiplier=8,
          kernel_size=(3,3),
          strides=1,
          padding='same',
          name='depthwise_conv2d_1',
          depthwise_initializer=self.weight_initializer
        ),
        tf.keras.layers.Reshape((8,88,1)),
        tf.keras.layers.BatchNormalization(momentum=0.9),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(
          pool_size=(3,3),
          strides=2,
          name='max_pool_2',
          padding='same'
        ),
        tf.keras.layers.Reshape((1,4,44)),
        tf.keras.layers.DepthwiseConv2D(
          depth_multiplier=8,
          kernel_size=(3,3),
          strides=1,
          padding='same',
          name='depthwise_conv2d_2',
          depthwise_initializer=self.weight_initializer
        ),
        tf.keras.layers.BatchNormalization(momentum=0.9),
        tf.keras.layers.ReLU(),
        tf.keras.layers.DepthwiseConv2D(
          depth_multiplier=16,
          kernel_size=(3,3),
          strides=1,
          padding='same',
          name='depthwise_conv2d_3',
          depthwise_initializer=self.weight_initializer
        ),
        tf.keras.layers.Reshape((4,5632,1)),
        tf.keras.layers.BatchNormalization(momentum=0.9),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(
          pool_size=(3,3),
          strides=2,
          name='max_pool_3',
          padding='same'
        ),
        tf.keras.layers.Reshape((1,2,2816)),
        tf.keras.layers.DepthwiseConv2D(
          depth_multiplier=16,
          kernel_size=(3,3),
          strides=1,
          padding='same',
          name='depthwise_conv2d_4',
          depthwise_initializer=self.weight_initializer
        ),
        tf.keras.layers.Reshape((2,45056,1)),
        tf.keras.layers.BatchNormalization(momentum=0.9),
        tf.keras.layers.ReLU(),
        tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first'),
        tf.keras.layers.Dense(1, kernel_initializer=self.weight_initializer)
      ], name='conv2d_block')
    else: self.conv2d_block = None

    if self.dwt:
        self.dwt_layer = WaveTFFactory().build(
          kernel_type='haar',
          dim=1,
        )
    else: self.dwt_layer = None

    self.classifier = tf.keras.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(
        self.num_feats * 32, 
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
    
    descriptors = []

    if self.eeg:
      eeg_conv = self.conv1d_block(inputs)
      descriptors.append(eeg_conv)

    if self.stft:
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
      stft_conv = self.conv2d_block(stft_out)
      descriptors.append(stft_conv)
    
    if self.fft:
      fft_out = tf.cast(tf.signal.fft(tf.cast(inputs, tf.complex64)), tf.float32)
      fft_conv = self.conv1d_block(fft_out)
      descriptors.append(fft_conv)
    
    if self.dwt:
      dwt_out = self.dwt_layer(inputs)
      dwt_conv = self.conv1d_block(tf.reshape(dwt_out, [-1, 178, 1]))
      descriptors.append(dwt_conv)

    concat = tf.concat(descriptors, axis=1)
    return self.classifier(concat)

  def get_config(self):
    super_config = super().get_config()
    super_config.update({
      'conv1d_block': self.conv1d_block,
      'conv2d_block': self.conv2d_block,
      'dwt_layer': self.dwt_layer,
      'classifier': self.classifier,
      'eeg': self.eeg,
      'fft': self.fft,
      'dwt': self.dwt,
      'stft': self.stft
    })
    return super_config
  
  @classmethod
  def from_config(cls, config):
    return cls(**config)
  