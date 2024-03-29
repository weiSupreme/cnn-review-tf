import tensorflow as tf
import os
from model_fn import model_fn, RestoreMovingAverageHook
from input_pipeline import Pipeline
tf.logging.set_verbosity('INFO')
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
"""
The purpose of this script is to train a network.
Evaluation will happen periodically.

To use it just run:
python train.py

Parameters below is for training 0.5x version.
"""

# 1281144/128 = 10008.9375
# so 1 epoch ~ 10000 steps

GPU_TO_USE = '3'
BATCH_SIZE = 128
VALIDATION_BATCH_SIZE = 32
NUM_EPOCHS = 50  # set 166 for 1.0x version
TRAIN_DATASET_SIZE = 60000
NUM_STEPS = NUM_EPOCHS * (TRAIN_DATASET_SIZE // BATCH_SIZE)
PARAMS = {
    'train_dataset_path': 'data/tiny_imagenet_train/',
    'val_dataset_path': 'data/tiny_imagenet_val/',
    'weight_decay': 0.01, # 4e-5,
    'initial_learning_rate': 0.0625,  #0.0625,  # 0.5/8
    'decay_steps': NUM_STEPS,
    'end_learning_rate': 1e-7,
    'model_dir': 'models/mnist_LeNet5',
    'num_classes': 10,
    'depth_multiplier': '0.5'  # set '1.0' for 1.0x version
}

mnist = input_data.read_data_sets('MNIST_data/', one_hot=False)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images},
    y={'labels':mnist.train.labels.astype(np.int32)},
    num_epochs=None,
    shuffle=True,
    batch_size=BATCH_SIZE)

val_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images},
    y={'labels':mnist.test.labels.astype(np.int32)},
    batch_size=VALIDATION_BATCH_SIZE,
    shuffle=False)

session_config = tf.ConfigProto(allow_soft_placement=True)
session_config.gpu_options.visible_device_list = GPU_TO_USE
session_config.gpu_options.allow_growth = True
run_config = tf.estimator.RunConfig()
run_config = run_config.replace(model_dir=PARAMS['model_dir'],
                                session_config=session_config,
                                save_summary_steps=500,
                                save_checkpoints_steps=1000,
                                log_step_count_steps=50)

estiamtor = tf.estimator.Estimator(model_fn=model_fn,
                                   params=PARAMS,
                                   config=run_config)

train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=NUM_STEPS)
eval_spec = tf.estimator.EvalSpec(
    val_input_fn,
    steps=None,
    throttle_secs=10,
    hooks=[RestoreMovingAverageHook(PARAMS['model_dir'])])

tf.estimator.train_and_evaluate(estiamtor, train_spec, eval_spec)
