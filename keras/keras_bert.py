# This example is heavily based on the idea from Jacob Zweig from
# https://towardsdatascience.com/bert-in-keras-with-tensorflow-hub-76bcbc9417b

import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
import numpy as np
from bert.tokenization import FullTokenizer
from tensorflow.keras import backend as K
import pickle

# Initialize session
sess = tf.Session()


def load_directory_data(directory):
  # Load all files from a directory in a DataFrame.
  data = {}
  data["sentence"] = []
  data["sentiment"] = []
  for file_path in os.listdir(directory):
    with tf.io.gfile.GFile(os.path.join(directory, file_path), "r") as f:
      data["sentence"].append(f.read())
      data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
  return pd.DataFrame.from_dict(data)


def load_dataset(directory):
  # Merge positive and negative examples, add a polarity column and shuffle.
  pos_df = load_directory_data(os.path.join(directory, "pos"))
  neg_df = load_directory_data(os.path.join(directory, "neg"))
  pos_df["polarity"] = 1
  neg_df["polarity"] = 0
  return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


def download_and_load_datasets(force_download=False):
  # Download and process the dataset files.
  dataset = tf.keras.utils.get_file(
      fname="aclImdb.tar.gz",
      origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
      extract=True,
  )

  train_df = load_dataset(os.path.join(
      os.path.dirname(dataset), "aclImdb", "train"))
  test_df = load_dataset(os.path.join(
      os.path.dirname(dataset), "aclImdb", "test"))

  return train_df, test_df


class BertInput(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, text, label=None):
    """Constructs a InputExample.
    Args:
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.text = text
    self.label = label


def create_tokenizer_from_hub_module(bert_path):
  """Get the vocab file and casing info from the Hub module."""
  bert_module = hub.Module(bert_path)
  tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
  vocab_file, do_lower_case = sess.run(
      [tokenization_info["vocab_file"], tokenization_info["do_lower_case"]]
  )
  return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def convert_single_example(tokenizer, example, max_seq_length=256):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  tokens = tokenizer.tokenize(example.text)
  tokens = tokens[0: (max_seq_length - 2)
                  ] if len(tokens) > max_seq_length - 2 else tokens

  tokens = ["[CLS]"] + tokens + ["[SEP]"]
  segment_ids = [0]*max_seq_length
  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)
  padlength = max_seq_length - len(input_ids)
  input_ids = input_ids + [0]*padlength
  input_mask = input_mask + [0]*padlength

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  return input_ids, input_mask, segment_ids, example.label


def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  input_ids, input_masks, segment_ids, labels = [], [], [], []
  print("Converting examples to features")
  for example in examples:
    input_id, input_mask, segment_id, label = convert_single_example(
        tokenizer, example, max_seq_length
    )
    input_ids.append(input_id)
    input_masks.append(input_mask)
    segment_ids.append(segment_id)
    labels.append(label)
  return (
      np.array(input_ids),
      np.array(input_masks),
      np.array(segment_ids),
      np.array(labels).reshape(-1, 1),
  )


def convert_text_to_examples(texts, labels):
  """Create InputExamples"""
  InputExamples = []
  for text, label in zip(texts, labels):
    InputExamples.append(
        BertInput(text=" ".join(text), label=label)
    )
  return InputExamples


class BertLayer(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    self.trainable = True
    self.output_size = 768
    self.pooling = "first"
    self.bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    super(BertLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    self.bert = hub.Module(
        self.bert_path, trainable=self.trainable, name="bert_module")

    # Remove unused layers
    trainable_vars = self.bert.variables
    trainable_vars = [
        var for var in trainable_vars if not "/cls/" in var.name]
    trainable_layers = ["pooler/dense"]

    # Update trainable vars to contain only the specified layers
    trainable_vars = [var for var in trainable_vars if any(
        [l in var.name for l in trainable_layers])]

    # Add to trainable weights
    for var in trainable_vars:
      self._trainable_weights.append(var)

    for var in self.bert.variables:
      if var not in self._trainable_weights:
        self._non_trainable_weights.append(var)

    super(BertLayer, self).build(input_shape)

  def call(self, inputs):
    inputs = [K.cast(x, dtype="int32") for x in inputs]
    input_ids, input_mask, segment_ids = inputs
    bert_inputs = dict(
        input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
    )
    pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
        "pooled_output"
    ]
    return pooled

  def compute_output_shape(self, input_shape):
    return (input_shape[0], self.output_size)


# Build model
def build_model(max_seq_length):
  in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
  in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
  in_segment = tf.keras.layers.Input(
      shape=(max_seq_length,), name="segment_ids")
  bert_inputs = [in_id, in_mask, in_segment]

  bert_output = BertLayer()(bert_inputs)
  dense = tf.keras.layers.Dense(256, activation="relu")(bert_output)
  pred = tf.keras.layers.Dense(1, activation="sigmoid")(dense)

  model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
  model.compile(loss="binary_crossentropy",
                optimizer="adam", metrics=["accuracy"])
  model.summary()

  return model


def initialize_vars(sess):
  sess.run(tf.local_variables_initializer())
  sess.run(tf.global_variables_initializer())
  sess.run(tf.tables_initializer())
  K.set_session(sess)


def main():
  # Params for bert model and tokenization
  bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
  max_seq_length = 256

  train_df, test_df = download_and_load_datasets()

  # Create datasets (Only take up to max_seq_length words for memory)
  train_text = train_df["sentence"].tolist()
  train_text = [" ".join(t.split()[0:max_seq_length]) for t in train_text]
  train_text = np.array(train_text, dtype=object)[:, np.newaxis]
  train_label = train_df["polarity"].tolist()

  test_text = test_df["sentence"].tolist()
  test_text = [" ".join(t.split()[0:max_seq_length]) for t in test_text]
  test_text = np.array(test_text, dtype=object)[:, np.newaxis]
  test_label = test_df["polarity"].tolist()

  # Instantiate tokenizer
  tokenizer = create_tokenizer_from_hub_module(bert_path)

  # Convert data to InputExample format
  train_examples = convert_text_to_examples(train_text, train_label)
  test_examples = convert_text_to_examples(test_text, test_label)

  # Convert to features
  (
      train_input_ids,
      train_input_masks,
      train_segment_ids,
      train_labels,
  ) = convert_examples_to_features(
      tokenizer, train_examples, max_seq_length=max_seq_length
  )
  (
      test_input_ids,
      test_input_masks,
      test_segment_ids,
      test_labels,
  ) = convert_examples_to_features(
      tokenizer, test_examples, max_seq_length=max_seq_length
  )

  model = build_model(max_seq_length)

  # Instantiate variables
  initialize_vars(sess)

  model.fit(
      [train_input_ids, train_input_masks, train_segment_ids],
      train_labels,
      validation_data=(
          [test_input_ids, test_input_masks, test_segment_ids],
          test_labels,
      ),
      epochs=1,
      batch_size=32,
  )
  model.save("bert_keras.h5")


if __name__ == "__main__":
  main()
