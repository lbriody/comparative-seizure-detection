import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser, Namespace
from collections import defaultdict


def one_hot_encode(
  labels: np.ndarray,
  seizure_class: int
) -> np.ndarray:
  labels[labels != seizure_class] = 0 # not seizure
  if seizure_class != 1: labels[labels == seizure_class] = 1 # seizure
  return labels.astype(np.int32)


def train_test_split_csv(
  path,
  num_cols,
  seizure_class,
  input_dtype=np.float32,
  label_dtype=np.int32,
  has_headers=True,
  test_size=0.2,
  random_state=42
) -> None:
  # set datatypes for cols
  dtypes = defaultdict(lambda: input_dtype)
  dtypes[num_cols-1] = label_dtype # last col is int
  
  # open csv
  if has_headers:
    data = pd.read_csv(
      path,
      usecols=[i+1 for i in range(0, num_cols)],
      header=0,
      dtype=dtypes
    )
  else:
    data = pd.read_csv(
      path,
      usecols=[i+1 for i in range(0, num_cols)],
      header=None,
      dtype=dtypes
    )

  # split into inputs and labels (labels in last column)
  inputs = data.iloc[:, :-1]
  labels = data.iloc[:, -1]

  # split into train and test sets
  X0, X1, Y0, Y1 = train_test_split(
    inputs,
    labels,
    test_size=test_size,
    random_state=random_state,
    shuffle=True
  )

  # get data directory from path
  data_dir = os.path.dirname(path)

  # make train and test subdirs of data directory
  os.makedirs(
    os.path.join(
      data_dir,
      'train'
    ),
    exist_ok=True
  )
  os.makedirs(
    os.path.join(
      data_dir,
      'test'
    ),
    exist_ok=True
  )

  # save train and test sets to .npy files in train and test subdirs of data dir
  np.save(
    os.path.join(
      data_dir, 
      'train', 
      'inputs.npy'
    ), 
    X0
  )
  np.save(
    os.path.join(
      data_dir,
      'train',
      'labels.npy'
    ),
    one_hot_encode(Y0, seizure_class)
  )
  np.save(
    os.path.join(
      data_dir,
      'test',
      'inputs.npy'
    ),
    X1
  )
  np.save(
    os.path.join(
      data_dir,
      'test',
      'labels.npy'
    ),
    one_hot_encode(Y1, seizure_class)
  )


def parse_args(args=None) -> Namespace:
  parser = ArgumentParser(description='Split data into train and test sets.')
  parser.add_argument(
    '--path',
    type=str,
    default='data/preprocessed-uci-eeg.csv',
    help='path to data csv.'
  )
  parser.add_argument(
    '--seizure_class',
    type=int,
    default=1,
    help='class to classify as seizure.'
  )
  parser.add_argument(
    '--num_cols',
    type=int,
    default=179,
    help='number of columns in csv (excluding index col 0).'
  )
  parser.add_argument(
    '--input_dtype',
    type=str,
    default='float32',
    help='dtype of input columns.'
  )
  parser.add_argument(
    '--label_dtype',
    type=str,
    default='int32',
    help='dtype of label column.'
  )
  parser.add_argument(
    '--no_headers',
    action='store_false',
    help='optional flag to indicate csv has no header row.'
  )
  parser.add_argument(
    '--test_size',
    type=float,
    default=0.2,
    help='size of test set.'
  )
  parser.add_argument(
    '--random_state',
    type=int,
    default=42,
    help='random state.'
  )
  
  if args is None:
    return parser.parse_args()   # For calling through command line/terminal.
  return parser.parse_args(args) # For calling through notebook.


def main(args):
  # convert dtypes from str to np.dtype
  input_dtype = np.dtype(getattr(np, args.input_dtype))
  label_dtype = np.dtype(getattr(np, args.label_dtype))

  if args.no_headers:
    train_test_split_csv(
      path=args.path,
      seizure_class=args.seizure_class,
      num_cols=args.num_cols,
      input_dtype=input_dtype,
      label_dtype=label_dtype,
      test_size=args.test_size,
      random_state=args.random_state
    )
  else:
    train_test_split_csv(
      path=args.path,
      seizure_class=args.seizure_class,
      num_cols=args.num_cols,
      input_dtype=input_dtype,
      label_dtype=label_dtype,
      has_headers=False,
      test_size=args.test_size,
      random_state=args.random_state
    )


if __name__ == '__main__':
  main(parse_args())
  