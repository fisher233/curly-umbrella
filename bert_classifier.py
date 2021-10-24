# -*- coding: utf-8 -*-
"""
Created on Sat Aug 03 13:58:24 2019

@author: 23338
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from bert import tokenization
from bert import modeling
from bert import optimization
import os
import pandas as pd

class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self, data_dir):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

#  @classmethod
#  def _read_tsv(cls, input_file, quotechar=None):
#    """Reads a tab separated value file."""
#    with tf.gfile.Open(input_file, "r") as f:
#      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
#      lines = []
#      for line in reader:
#        lines.append(line)
#      return lines

class DataFrameProcessor(DataProcessor):

    def get_train_examples(self, data_dir, file_name='train.csv', data_col = 'text', label_col = 'label'):
        df = pd.read_csv(os.path.join(data_dir, file_name))
        X = df[data_col].values
        y = df[label_col].values
        return self.create_examples(X, y)

    def get_dev_examples(self, data_dir, file_name='dev.csv', data_col = 'text', label_col = 'label'):
        return self.get_train_examples(data_dir, file_name, data_col, label_col)

    def get_test_examples(self, data_dir, file_name='test.csv', data_col = 'text'):
        return self.create_examples(pd.read_csv(os.path.join(data_dir, file_name))[data_col].values)

    def get_labels(self, data_dir, file_name='labels.csv', squeeze=True):
        label_list = pd.read_csv(os.path.join(data_dir, file_name), squeeze=squeeze).values
        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i
        return label_list, label_map
    
    @classmethod
    def create_examples(cls, X, y = None, set_type = 'train'):
        examples = []
        for (i, x) in enumerate(X):
            if y is not None:
                examples.append(InputExample(guid=None, text_a=x, label = y[i]))
            else:
                examples.append(InputExample(guid=None, text_a=x))
        return examples

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example
    
    
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def convert_single_example(ex_index, example, label_map, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

#  label_map = {}
#  for (i, label) in enumerate(label_list):
#    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = None
  if example.label:
      label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    if example.label:
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature

def convert_examples_to_features(examples, label_map, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_map,  max_seq_length, tokenizer)
        features.append(feature)
    return features

def create_model(bert_config, mode, input_ids, input_mask, segment_ids, num_labels, labels = None):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    model = modeling.BertModel(config = bert_config, 
                       is_training = is_training, 
                       input_ids = input_ids,
                       input_mask = input_mask,
                       token_type_ids=segment_ids,
                       use_one_hot_embeddings=False) # use one-hot method or tf.gather() for word embeddings. 
    
    cls_output = model.get_pooled_output() # pretrain output/embedding, the output of [CLS] token
    hidden_size = cls_output.shape[-1].value
    output_weights = tf.get_variable(name='output_weights', 
                                     shape = [num_labels, hidden_size], # [hidden_size, num_labels]
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable(name='output_bias', 
                                  shape = [num_labels],
                                  initializer=tf.zeros_initializer())
    if is_training:
        cls_output = tf.nn.dropout(cls_output, keep_prob=0.9) # tf.layers.dropout(cls_output, rate=0.1)
    logits = tf.matmul(cls_output, output_weights, transpose_b=True) # no transpose if [hidden_size, num_labels]
    logits = tf.nn.bias_add(logits, output_bias) # tf.add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1) # axis=-1 is the default if None

    if mode == tf.estimator.ModeKeys.PREDICT:
        predicted_labels = tf.squeeze(tf.argmax(probabilities, axis=-1, output_type=tf.int32)) # remove dimensions of size 1

        return (predicted_labels, probabilities)
    else:
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(labels, num_labels, dtype=tf.float32)
        loss_per_examples = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(loss_per_examples)
        return (loss, probabilities)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate=None, num_train_steps=None, num_warmup_steps=None):
    def model_fn(features, labels, mode): # params, config
        '''features, labels: a single tf.Tensor or dictionary of tf.Tensor. If mode is PREDICT, 
           labels = None will be received.
        '''
        
        input_ids = features['input_ids']
        input_mask = features['input_mask']
        segment_ids = features['segment_ids']
#        label_ids = features["label_ids"]
        
        estimatorSpec = None
        if mode == tf.estimator.ModeKeys.PREDICT:
            predicted_labels, probabilities = create_model(bert_config, mode, input_ids, input_mask, segment_ids, num_labels)
            estimatorSpec = tf.estimator.EstimatorSpec(mode=mode, predictions={"predicted_labels":predicted_labels, 
                                                                               "probabilities": probabilities})
        else:
            loss, probabilities = create_model(bert_config, mode, input_ids, input_mask, segment_ids, num_labels, labels)
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_op = optimization.create_optimizer(loss, learning_rate, num_train_steps, num_warmup_steps, False)
                estimatorSpec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
            else:
                def metric_fn(labels, probabilities):
                    predicted_labels = tf.argmax(probabilities, axis=-1, output_type=tf.int32)
                    tf.logging.info("  name = %s, shape = %s", labels.shape, predicted_labels.shape)
                    return {
                            "accuracy":tf.metrics.accuracy(labels, predicted_labels)
                            }
                
                eval_metrics = metric_fn(labels, probabilities)
                estimatorSpec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops =  eval_metrics)
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)
        return estimatorSpec
    
    return model_fn

def input_fn_builder(features, seq_length, is_training, drop_remainder, batch_size, shuffle_buffer_size=None, seed=None):
  """Creates an `input_fn` closure to be passed to Estimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    if feature.label_id is not None:
        all_label_ids.append(feature.label_id)

  def input_fn(): # params
    """The actual input function."""
#    batch_size = params["batch_size"]

    num_examples = len(features)
    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    if len(all_label_ids) == num_examples:
        d = tf.data.Dataset.from_tensor_slices(({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32)
    #        "label_ids":
    #            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32)
        }, tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32)))
    else:
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32)
        })

    if is_training:
        d = d.repeat()
        if shuffle_buffer_size is None:
            d = d.shuffle(buffer_size=num_examples, seed=seed)
        else:
            d = d.shuffle(buffer_size=shuffle_buffer_size, seed=seed) # 100

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn