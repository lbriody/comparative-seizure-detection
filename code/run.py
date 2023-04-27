import tensorflow as tf
from tensorflow import keras
import tensorboard
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from argparse import ArgumentParser, Namespace

# import models
from models.fourier import FourierModel 
from models.lstm_conv1d import LSTMConvModel
from models.lstm_optical import LSTMOpticalModel


def train(args, model: tf.keras.Model) -> None:
  # load full data from csv
  X0 = np.load(
    os.path.join(
      args.data, 
      'train', 
      'inputs.npy'
    )
  )
  Y0 = np.load(
    os.path.join(
      args.data, 
      'train', 
      'labels.npy'
    )
  )
  X1 = np.load(
    os.path.join(
      args.data, 
      'test', 
      'inputs.npy'
    )
  )
  Y1 = np.load(
    os.path.join(
      args.data, 
      'test', 
      'labels.npy'
    )
  )

  # create log and ckpt directories
  log_dir = os.path.join(
    'logs',
    args.model,
    datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
  )
  os.makedirs(
    log_dir,
    exist_ok=True
  )
  checkpoint_path = os.path.join(
    'checkpoints',
    args.model,
    datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
  )
  os.makedirs(
    checkpoint_path,
    exist_ok=True
  )

  # create callbacks
  tensorboard_cb = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, 
    histogram_freq=1
  )
  ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path + 'cp-{epoch:04d}-{val_accuracy:04d}-{val_loss:04d}.ckpt',
    monitor='val_accuracy',
    save_weights_only=True,
    verbose=1
  )
  earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10
  )
  reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=1,
    factor= 0.9,
    mode='max',
    verbose=0
  )
  terminate_nan_cb = tf.keras.callbacks.TerminateOnNaN()
  cbs = [
    tensorboard_cb,
    ckpt_cb,
    earlystop_cb,
    reduce_lr_cb,
    terminate_nan_cb
  ]

  # create model
  model.compile(
    optimizer=tf.keras.optimizers.Adam(
      learning_rate=args.lr
    ),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
      tf.keras.metrics.BinaryAccuracy(),
      tf.keras.metrics.Precision(),
      tf.keras.metrics.Recall()
    ]
  )

  # train model
  model.fit(
    x=X0,
    y=Y0,
    validation_data=(X1, Y1),
    batch_size=args.batch_size,
    epochs=args.epochs,
    callbacks=cbs
  )    


def test(args, model: tf.keras.Model) -> None:
  X1 = np.load(
    os.path.join(
      args.data, 
      'test', 
      'inputs.npy'
    )
  )
  Y1 = np.load(
    os.path.join(
      args.data, 
      'test', 
      'labels.npy'
    )
  )

  model.compile(
    optimizer=tf.keras.optimizers.Adam(
      learning_rate=args.lr
    ),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
      tf.keras.metrics.BinaryAccuracy(),
      tf.keras.metrics.Precision(),
      tf.keras.metrics.Recall()
    ]
  )

  model.evaluate(
    x=X1,
    y=Y1,
    batch_size=args.batch_size
  )


def get_model(args) -> keras.Model:
  if args.model == 'lstm_conv':
    return LSTMConvModel()
  elif args.model == 'lstm_optical':
    return LSTMOpticalModel()
  elif args.model == 'fourier':
    return FourierModel()


def parse_args(args=None) -> Namespace:
  parser = ArgumentParser(description='train and test models.')
  parser.add_argument(
    '--load_weights',
    type=str, 
    default=None, 
    help='path to model weights.'
  )
  parser.add_argument(
    '--task', 
    type=str, 
    default='train', 
    choices=['train', 'test', 'both'], 
    help='task to run.'
  )
  parser.add_argument(
    '--data', 
    type=str, 
    default='data', 
    help='path to data directory.'
  )
  parser.add_argument(
    '--model', 
    type=str, 
    default='fourier', 
    choices=['1','2','3','fourier','lstm_conv1d','lstm_optical'], 
    help='model to use.'
  )
  parser.add_argument(
    '--epochs', 
    type=int, 
    default=50, 
    help='number of epochs to train.'
  )
  parser.add_argument(
    '--batch_size', 
    type=int, 
    default=6, 
    help='batch size.'
  )
  parser.add_argument(
    '--lr', 
    type=float, 
    default=0.005, 
    help='learning rate.'
  )
  
  if args is None:
    return parser.parse_args()   # For calling through command line/terminal.
  return parser.parse_args(args) # For calling through notebook.

def main(args) -> None:
  model = get_model(args)
  if args.load_weights is not None: model.load_weights(args.load_weights)

  if args.task in ('train', 'both'): 
    train(args, model)

  if args.task == 'test':
    if args.load_weights is None: raise ValueError('load_weights must be specified for testing.')
    test(args, model)
  
  if args.task == 'both': 
    test(args, model)


if __name__ == '__main__':
  main(parse_args())
