# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import json
import re

import modeling
import tokenization
import tensorflow as tf
import feature_extraction_lib

import numpy as np
from sklearn.cluster import KMeans
from nltk.tokenize.moses import MosesDetokenizer

flags = tf.flags

FLAGS = flags.FLAGS

# Flags that were defined in the origin BERT code.
flags.DEFINE_string("input_file", None, "")

flags.DEFINE_string("output_file", None, "")

flags.DEFINE_integer("layer_index", -1, "By default, use last layer.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("batch_size", 32, "Batch size for predictions.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("master", None,
                    "If using a TPU, the address of the master.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "use_one_hot_embeddings", False,
    "If True, tf.one_hot will be used for embedding lookups, otherwise "
    "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
    "since it is much faster.")

# Custom FLAGS for our purposes.
flags.DEFINE_integer("num_clusters", 10, 
    "The number of clusters for k-means.")


def read_examples_one_per_line(input_file):
  """Read a list of `InputExample`s from an input file.
  Returns alist of InputExample objects.
  """
  # This one isn't get called anywhere currently.
  examples = []
  unique_id = 0
  with tf.gfile.GFile(input_file, "r") as reader:
    while True:
      line = tokenization.convert_to_unicode(reader.readline())
      if not line:
        break
      line = line.strip()
      text_a = None
      text_b = None
      m = re.match(r"^(.*) \|\|\| (.*)$", line)
      if m is None:
        text_a = line
      else:
        text_a = m.group(1)
        text_b = m.group(2)
      examples.append(feature_extraction_lib.InputExample(
          unique_id=unique_id, text_a=text_a, text_b=text_b))
      unique_id += 1
  return examples

def read_examples_json(input_file):
  """Read a list of `InputExample`s from an input file.
  Returns a list of lists of InputExample objects (one list for each example
  in the json.) 
  """
  detokenizer = MosesDetokenizer()

  examples = []
  examples_indices = []

  unique_id = 0
  with tf.gfile.GFile(input_file, "r") as reader:
    data = json.load(reader)
    for example_index, example_data in enumerate(data):
      for line in example_data['pred']:
        line = detokenizer.detokenize(line, return_str=True)
        examples.append(feature_extraction_lib.InputExample(
            unique_id=unique_id, text_a=line, text_b=None))
        examples_indices.append(example_index)
        unique_id += 1
  return examples, examples_indices, data

def get_features():
  """Reads in the output of OpenNMT-py decoding and outputs features for each sequence.

  Returns:
    A list containing one entry for each example in the OpenNMT-py json output. Each 
    entry consists of a list with an entry for each possible response to that example 
    query.
  """
  layer_index = FLAGS.layer_index

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      master=FLAGS.master,
      tpu_config=tf.contrib.tpu.TPUConfig(
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  examples, unique_id_to_example_idx, input_data = read_examples_json(FLAGS.input_file)

  # for example_list in examples:
  features = feature_extraction_lib.convert_examples_to_features(
      examples=examples,
      seq_length=FLAGS.max_seq_length,
      tokenizer=tokenizer)

  unique_id_to_feature = {}
  for feature in features:
    unique_id_to_feature[feature.unique_id] = feature

  model_fn = feature_extraction_lib.model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      layer_indexes=[layer_index],
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      predict_batch_size=FLAGS.batch_size)

  input_fn = feature_extraction_lib.input_fn_builder(
      features=features, seq_length=FLAGS.max_seq_length)

  # with codecs.getwriter("utf-8")(tf.gfile.Open(FLAGS.output_file,
                                               # "w")) as writer:
  all_results = [[] for _ in range(1+max(unique_id_to_example_idx))]
  for result in estimator.predict(input_fn, yield_single_examples=True):
    unique_id = int(result["unique_id"])
    feature = unique_id_to_feature[unique_id]
    example_index = unique_id_to_example_idx[unique_id]

    layer_output = result["layer_output_0"]
    example_dict = {}
    example_dict['tokens'] = feature.tokens
    example_dict['values'] = layer_output[:len(feature.tokens), :]
    example_dict['unique_id'] = unique_id

    resp_index = len(all_results[example_index])
    example_dict['score'] = input_data[example_index]['scores'][resp_index]
    example_dict['pred'] = input_data[example_index]['pred'][resp_index]

    print('Adding an example for example index %d' % (example_index))
    all_results[example_index].append(example_dict)
  return all_results
 
def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  features = get_features()
  for example_idx, example in enumerate(features):
    response_embeddings = []
    for response in example:
      # response_emb = np.mean(response['values'], axis=0)
      response_emb = response['values'][0]
      response_embeddings.append(response_emb)
    kmeans = KMeans(n_clusters=FLAGS.num_clusters).fit(
        response_embeddings)

    print('===EXAMPLE %d===' % (example_idx))
    for cluster_idx in range(FLAGS.num_clusters):
      print('Cluster %d: ' % cluster_idx)
      r_in_cluster = [r for r, c in zip(example, kmeans.labels_) if c == cluster_idx]

      # Output all of the responses in the cluster, sorted hy likelihood.
      r_in_cluster = sorted(r_in_cluster, key=lambda r: r['score'])
      for response in r_in_cluster:
        print('  (%f): %s' % (response['score'], ' '.join(response['tokens'])))
      print('')


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("init_checkpoint")
  flags.mark_flag_as_required("output_file")
  tf.app.run()
