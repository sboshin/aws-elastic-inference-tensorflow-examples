import pandas as pd
import tensorflow_hub as hub
import os
import re
import numpy as np
from bert.tokenization import FullTokenizer
from tensorflow.keras import backend as K
import pickle
import tensorflow as tf


os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
# c5.4xlarge = 8, c5.18xlarge = 36, m5.4xlarge = 8
os.environ["OMP_NUM_THREADS"] = "8"

# Initialize session
sess = K.get_session()


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
  tokenization_info = bert_module(
      signature="tokenization_info", as_dict=True)
  sess.run(tf.global_variables_initializer())
  vocab_file, do_lower_case = sess.run(
      [tokenization_info["vocab_file"], tokenization_info["do_lower_case"]]
  )

  return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def convert_single_example(tokenizer, example, max_seq_length=256):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  tokens = tokenizer.tokenize(example.text)
  #tokens = tokenizer.tokenize(example)

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
  #return input_ids, input_mask, segment_ids, 0


def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  input_ids, input_masks, segment_ids = [], [], []
  for example in examples:
    input_id, input_mask, segment_id, _ = convert_single_example(
        tokenizer, example, max_seq_length
    )
    input_ids.append(input_id)
    input_masks.append(input_mask)
    segment_ids.append(segment_id)

  return (
      np.array(input_ids),
      np.array(input_masks),
      np.array(segment_ids),
  )


def convert_text_to_examples(texts):
  """Create InputExamples"""
  return [BertInput(text=" ".join(text)) for text in texts]


class BertLayer(tf.keras.layers.Layer):
  def get_config(self):
    config = {
        'pooling': self.pooling,
        'bert_path': self.bert_path,
    }
    base_config = super(BertLayer, self).get_config()
    config.update(base_config)
    return config
  def __init__(
      self,
      pooling="first",
      bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
      **kwargs
  ):    
    self.trainable = True
    self.output_size = 768
    self.pooling = "first"
    self.bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    super(BertLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    self.bert = hub.Module(
        self.bert_path, trainable=self.trainable, name="bert_module"
    )

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


def main():
  # Params for bert model and tokenization
  bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
  max_seq_length = 256

  # Instantiate tokenizer
  tokenizer = create_tokenizer_from_hub_module(bert_path)

  # Convert data to InputExample format
  #train_examples = convert_text_to_examples(train_text, train_label)
  test_text = [
      ["""
      Airplane! is still a good comedy film but it is compromised by being a topical film. Back in 1980, this was a very funny movie but in 2014, it doesn't hold up quite as well since disco, jive talking, the glorification of drug use and deadpan comedy aren't exactly in nowadays. A movie like The Jerk still holds up in 2014 because it doesn't rely on the pop culture of its time.
      That said, it's still very watchable and its screen time flies rapidly. You wish more movies had the genius idea of casting serious actors and playing off of their stoic style for laughs. Leslie Nielsen's career took off after playing Dr. Rumack and he had a successful back-nine when it came to acting. Julie Hagerty (in her debut) may come off as somewhat mousy and spaced-out as Elaine but she could also be a knockout as seen in the club scene.
      The movie is worth seeing for those classic exchanges and some funny spoofs but I just feel it doesn't have that punch that it once had after seeing it again recently.
      """],
       ["""
       I really tried to like this movie after reading the great reviews and hearing that it was one of the funniest films of all time over and over. But I just couldn't. I watched the whole movie without so much as cracking a smile. It wasn't that I didn't understand the jokes. On the contrary, I saw the point of every joke and why it was supposed to be funny, but I just didn't think it was humorous in the slightest. All told, I have watched this movie three times all the way through thinking maybe I had missed something or it would just "click" one time, but it didn't. I honestly have no idea why anyone would think that this is one of the greatest comedies ever or even think that it's funny.
       """],
  ]
  test_examples = convert_text_to_examples(test_text)
  for test in test_examples:
    print(test.text)

  (
      test_input_ids,
      test_input_masks,
      test_segment_ids,
  ) = convert_examples_to_features(
      tokenizer, test_examples, max_seq_length=max_seq_length
      #tokenizer, test_text, max_seq_length=max_seq_length
  )

  model = tf.keras.models.load_model(
      "bert_keras.h5", custom_objects={"BertLayer": BertLayer}, compile=False)
  ei_keras_on = False
  if(ei_keras_on):
    from tensorflow.contrib.ei.python.keras.ei_keras import EIKerasModel
    emodel = EIKerasModel(model)

  iterations = 10
  import time
  times = []
  for ii in range(iterations):
    start = time.time()
    if(ei_keras_on):
      emodel.predict(
          [test_input_ids, test_input_masks, test_segment_ids],
          batch_size=1,
      )
    else:
      a = model.predict(
          [test_input_ids, test_input_masks, test_segment_ids],
          batch_size=1,
      )
      print(a)
    end = time.time()
    times.append(end-start)
  timings = np.array(times)
  p50 = np.percentile(timings, 50, interpolation='nearest')
  p90 = np.percentile(timings, 90, interpolation='nearest')
  p99 = np.percentile(timings, 99, interpolation='nearest')

  print("P50: %f, P90: %f, P99: %f\n" % (p50, p90, p99))


if __name__ == "__main__":
  main()

