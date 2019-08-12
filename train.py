import tensorflow as tf
import os
from model_fn import model_fn, RestoreMovingAverageHook
from input_pipeline import Pipeline
tf.logging.set_verbosity('INFO')
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
BATCH_SIZE = 32
VALIDATION_BATCH_SIZE = 32
NUM_EPOCHS = 100  # set 166 for 1.0x version
TRAIN_DATASET_SIZE = 5000
NUM_STEPS = NUM_EPOCHS * (TRAIN_DATASET_SIZE // BATCH_SIZE)
PARAMS = {
    'train_dataset_path': 'data/imagenet10_train/',
    'val_dataset_path': 'data/imagenet10_val/',
    'weight_decay': 5e-5,  # 4e-5,
    'initial_learning_rate': 0.0625,  #0.0625,  # 0.5/8
    'decay_steps': NUM_STEPS,
    'end_learning_rate': 1e-7,
    'model_dir': 'models/imagenet10_mobilenetv3_small_0.33',
    'pretrained_dir':'',
    'num_classes': 10,
    'depth_multiplier': '0.5'  # set '1.0' for 1.0x version
}


def get_input_fn(is_training):

    dataset_path = PARAMS['train_dataset_path'] if is_training else PARAMS[
        'val_dataset_path']
    filenames = os.listdir(dataset_path)
    filenames = [n for n in filenames if n.endswith('.tfrecords')]
    filenames = [os.path.join(dataset_path, n) for n in sorted(filenames)]

    batch_size = BATCH_SIZE if is_training else VALIDATION_BATCH_SIZE
    num_epochs = None if is_training else 1

    def input_fn():
        pipeline = Pipeline(filenames,
                            is_training,
                            batch_size=batch_size,
                            num_epochs=num_epochs)
        return pipeline.dataset

    return input_fn


session_config = tf.ConfigProto(allow_soft_placement=True)
session_config.gpu_options.visible_device_list = GPU_TO_USE
session_config.gpu_options.allow_growth = True
run_config = tf.estimator.RunConfig()
run_config = run_config.replace(model_dir=PARAMS['model_dir'],
                                session_config=session_config,
                                save_summary_steps=500,
                                save_checkpoints_steps=1000,
                                log_step_count_steps=50)

train_input_fn = get_input_fn(is_training=True)
val_input_fn = get_input_fn(is_training=False)
estiamtor = tf.estimator.Estimator(model_fn=model_fn,
                                   params=PARAMS,
                                   config=run_config)

train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=NUM_STEPS)
eval_spec = tf.estimator.EvalSpec(
    val_input_fn,
    steps=None,
    throttle_secs=120,
    hooks=[RestoreMovingAverageHook(PARAMS['model_dir'])])

tf.estimator.train_and_evaluate(estiamtor, train_spec, eval_spec)
