# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Executes BERT benchmarks and accuracy tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os
import time

# pylint: disable=g-bad-import-order
from absl import flags
from absl.testing import flagsaver
import tensorflow as tf
# pylint: enable=g-bad-import-order

from official.bert import modeling
from official.bert import run_classifier
from official.bert.benchmark import benchmark_utils
from official.utils.misc import distribution_utils

# pylint: disable=line-too-long
PRETRAINED_CHECKPOINT_PATH = 'gs://cloud-tpu-checkpoints/bert/tf_20/uncased_L-24_H-1024_A-16/bert_model.ckpt'
CLASSIFIER_TRAIN_DATA_PATH = 'gs://tf-perfzero-data/bert/classification/mrpc_train.tf_record'
CLASSIFIER_EVAL_DATA_PATH = 'gs://tf-perfzero-data/bert/classification/mrpc_eval.tf_record'
CLASSIFIER_INPUT_META_DATA_PATH = 'gs://tf-perfzero-data/bert/classification/mrpc_meta_data'
MODEL_CONFIG_FILE_PATH = 'gs://cloud-tpu-checkpoints/bert/tf_20/uncased_L-24_H-1024_A-16/bert_config'
# pylint: enable=line-too-long

FLAGS = flags.FLAGS


class BertClassifyBenchmarkBase(benchmark_utils.BertBenchmarkBase):
  """Base class to hold methods common to test classes in the module."""

  def __init__(self, output_dir=None):
    self.num_epochs = None
    self.num_steps_per_epoch = None

    super(BertClassifyBenchmarkBase, self).__init__(output_dir)

  @flagsaver.flagsaver
  def _run_bert_classifier(self, callbacks=None):
    """Starts BERT classification task."""
    with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
      input_meta_data = json.loads(reader.read().decode('utf-8'))

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    epochs = self.num_epochs if self.num_epochs else FLAGS.num_train_epochs
    if self.num_steps_per_epoch:
      steps_per_epoch = self.num_steps_per_epoch
    else:
      train_data_size = input_meta_data['train_data_size']
      steps_per_epoch = int(train_data_size / FLAGS.train_batch_size)
    warmup_steps = int(epochs * steps_per_epoch * 0.1)
    eval_steps = int(
        math.ceil(input_meta_data['eval_data_size'] / FLAGS.eval_batch_size))
    strategy = distribution_utils.get_distribution_strategy(
        distribution_strategy='mirrored', num_gpus=self.num_gpus)

    run_classifier.run_customized_training(
        strategy,
        bert_config,
        input_meta_data,
        FLAGS.model_dir,
        epochs,
        steps_per_epoch,
        eval_steps,
        warmup_steps,
        FLAGS.learning_rate,
        FLAGS.init_checkpoint,
        custom_callbacks=callbacks)


class BertClassifyBenchmarkReal(BertClassifyBenchmarkBase):
  """Short benchmark performance tests for BERT model.

  Tests BERT classification performance in different GPU configurations.
  The naming convention of below test cases follow
  `benchmark_(number of gpus)_gpu_(dataset type)` format.
  """

  def __init__(self, output_dir=None, **kwargs):
    super(BertClassifyBenchmarkReal, self).__init__(output_dir=output_dir)

    self.train_data_path = CLASSIFIER_TRAIN_DATA_PATH
    self.eval_data_path = CLASSIFIER_EVAL_DATA_PATH
    self.bert_config_file = MODEL_CONFIG_FILE_PATH
    self.input_meta_data_path = CLASSIFIER_INPUT_META_DATA_PATH

    # Since we only care about performance metrics, we limit
    # the number of training steps and epochs to prevent unnecessarily
    # long tests.
    self.num_steps_per_epoch = 110
    self.num_epochs = 1

  def _run_and_report_benchmark(self,
                                training_summary_path,
                                min_accuracy=0,
                                max_accuracy=1):
    """Starts BERT performance benchmark test."""

    start_time_sec = time.time()
    self._run_bert_classifier(callbacks=[self.timer_callback])
    wall_time_sec = time.time() - start_time_sec

    with tf.io.gfile.GFile(training_summary_path, 'rb') as reader:
      summary = json.loads(reader.read().decode('utf-8'))

    # Since we do not load from any pretrained checkpoints, we ignore all
    # accuracy metrics.
    summary.pop('eval_metrics', None)
    super(BertClassifyBenchmarkReal, self)._report_benchmark(
        stats=summary,
        wall_time_sec=wall_time_sec,
        min_accuracy=min_accuracy,
        max_accuracy=max_accuracy)

  def benchmark_1_gpu_mrpc(self):
    """Test BERT model performance with 1 GPU."""

    self._setup()
    self.num_gpus = 1
    FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_mrpc')
    FLAGS.train_data_path = self.train_data_path
    FLAGS.eval_data_path = self.eval_data_path
    FLAGS.input_meta_data_path = self.input_meta_data_path
    FLAGS.bert_config_file = self.bert_config_file
    FLAGS.train_batch_size = 4
    FLAGS.eval_batch_size = 4

    summary_path = os.path.join(FLAGS.model_dir, 'training_summary.txt')
    self._run_and_report_benchmark(summary_path)

  def benchmark_2_gpu_mrpc(self):
    """Test BERT model performance with 2 GPUs."""

    self._setup()
    self.num_gpus = 2
    FLAGS.model_dir = self._get_model_dir('benchmark_2_gpu_mprc')
    FLAGS.train_data_path = self.train_data_path
    FLAGS.eval_data_path = self.eval_data_path
    FLAGS.input_meta_data_path = self.input_meta_data_path
    FLAGS.bert_config_file = self.bert_config_file
    FLAGS.train_batch_size = 8
    FLAGS.eval_batch_size = 8

    summary_path = os.path.join(FLAGS.model_dir, 'training_summary.txt')
    self._run_and_report_benchmark(summary_path)

  def benchmark_4_gpu_mrpc(self):
    """Test BERT model performance with 4 GPUs."""

    self._setup()
    self.num_gpus = 4
    FLAGS.model_dir = self._get_model_dir('benchmark_4_gpu_mrpc')
    FLAGS.train_data_path = self.train_data_path
    FLAGS.eval_data_path = self.eval_data_path
    FLAGS.input_meta_data_path = self.input_meta_data_path
    FLAGS.bert_config_file = self.bert_config_file
    FLAGS.train_batch_size = 16

    summary_path = os.path.join(FLAGS.model_dir, 'training_summary.txt')
    self._run_and_report_benchmark(summary_path)

  def benchmark_8_gpu_mrpc(self):
    """Test BERT model performance with 8 GPUs."""

    self._setup()
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_mrpc')
    FLAGS.train_data_path = self.train_data_path
    FLAGS.eval_data_path = self.eval_data_path
    FLAGS.input_meta_data_path = self.input_meta_data_path
    FLAGS.bert_config_file = self.bert_config_file

    summary_path = os.path.join(FLAGS.model_dir, 'training_summary.txt')
    self._run_and_report_benchmark(summary_path)


class BertClassifyAccuracy(BertClassifyBenchmarkBase):
  """Short accuracy test for BERT model.

  Tests BERT classification task model accuracy. The naming
  convention of below test cases follow
  `benchmark_(number of gpus)_gpu_(dataset type)` format.
  """

  def __init__(self, output_dir=None, **kwargs):
    self.train_data_path = CLASSIFIER_TRAIN_DATA_PATH
    self.eval_data_path = CLASSIFIER_EVAL_DATA_PATH
    self.bert_config_file = MODEL_CONFIG_FILE_PATH
    self.input_meta_data_path = CLASSIFIER_INPUT_META_DATA_PATH
    self.pretrained_checkpoint_path = PRETRAINED_CHECKPOINT_PATH

    super(BertClassifyAccuracy, self).__init__(output_dir=output_dir)

  def _run_and_report_benchmark(self,
                                training_summary_path,
                                min_accuracy=0.84,
                                max_accuracy=0.88):
    """Starts BERT accuracy benchmark test."""

    start_time_sec = time.time()
    self._run_bert_classifier(callbacks=[self.timer_callback])
    wall_time_sec = time.time() - start_time_sec

    with tf.io.gfile.GFile(training_summary_path, 'rb') as reader:
      summary = json.loads(reader.read().decode('utf-8'))

    super(BertClassifyAccuracy, self)._report_benchmark(
        stats=summary,
        wall_time_sec=wall_time_sec,
        min_accuracy=min_accuracy,
        max_accuracy=max_accuracy)

  def benchmark_8_gpu_mrpc(self):
    """Run BERT model accuracy test with 8 GPUs.

    Due to comparatively small cardinality of  MRPC dataset, training
    accuracy metric has high variance between trainings. As so, we
    set the wide range of allowed accuracy (84% to 88%).
    """

    self._setup()
    FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_mrpc')
    FLAGS.train_data_path = self.train_data_path
    FLAGS.eval_data_path = self.eval_data_path
    FLAGS.input_meta_data_path = self.input_meta_data_path
    FLAGS.bert_config_file = self.bert_config_file
    FLAGS.init_checkpoint = self.pretrained_checkpoint_path

    summary_path = os.path.join(FLAGS.model_dir, 'training_summary.txt')
    self._run_and_report_benchmark(summary_path)


if __name__ == '__main__':
  tf.test.main()
