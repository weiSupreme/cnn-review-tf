import tensorflow as tf
from architectures.MobileNetV3 import MobileNetV3 as cnn
from functools import reduce
from operator import mul

MOMENTUM = 0.9
USE_NESTEROV = True
MOVING_AVERAGE_DECAY = 0.995
RESIZE_SIZE = 224

global INIT_FLAG
INIT_FLAG = False


def model_fn(features, labels, mode, params):
    x = features['images']
    if mode == tf.estimator.ModeKeys.PREDICT:
        with tf.name_scope('image_preprocess'):
            x = tf.image.resize_images(x, [RESIZE_SIZE, RESIZE_SIZE], method=0)
            x = (1.0 / 255.0) * tf.to_float(x)
            x = tf.reshape(x, shape=[1, RESIZE_SIZE, RESIZE_SIZE, -1])
    else:
        with tf.name_scope('image_preprocess'):
            x = tf.reshape(x, shape=[-1, RESIZE_SIZE, RESIZE_SIZE, 3])
    with tf.name_scope('standardize_input'):
        x = (2.0 * x) - 1.0

    is_training = mode == tf.estimator.ModeKeys.TRAIN
    logits = cnn(x,
                 is_training,
                 num_classes=params['num_classes'],
                 depth_multiplier=params['depth_multiplier'])
    predicted_class = tf.argmax(logits, 1, output_type=tf.int32)
    predictions = {
        'class_ids': predicted_class[:, tf.newaxis],
        'probabilities': tf.nn.softmax(logits, axis=1),
        'logits': logits
    }

    with tf.name_scope('evaluation_ops'):
        accuracy = tf.metrics.accuracy(labels['labels'], predicted_class)
        top5_accuracy = tf.metrics.mean(
            tf.to_float(
                tf.nn.in_top_k(predictions=predictions['probabilities'],
                               targets=labels['labels'],
                               k=5)))

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    global INIT_FLAG
    init_fn = None
    if INIT_FLAG:
        exclude = [
            'global_step:0'
        ]  #,'classifier/biases:0','classifier/weights:0'] # ['ShuffleNetV2/Conv1/weights:0','global_step:0','classifier/biases:0','classifier/weights:0'
        #]
        all_variables = tf.contrib.slim.get_variables_to_restore()
        varialbes_to_use = []
        num_params = 0
        for v in all_variables:
            shape = v.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
            if v.name not in exclude:
                print(v.name)
                varialbes_to_use.append(v)
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(
            tf.train.latest_checkpoint(params['pretrained_dir']),
            varialbes_to_use,
            ignore_missing_vars=True)
        print('***********************params: ', num_params)

    with tf.name_scope('weight_decay'):
        add_weight_decay(params['weight_decay'])
        regularization_loss = tf.losses.get_regularization_loss()

    with tf.variable_scope('cross_entropy'):
        #compute loss
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels['labels'],
                                                      logits=logits)
        #loss = tf.reduce_mean(loss, axis=0)
        tf.losses.add_loss(loss)
    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {
            'val_accuracy': accuracy,
            'val_top5_accuracy': top5_accuracy
        }
        return tf.estimator.EstimatorSpec(mode,
                                          loss=total_loss,
                                          eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN
    with tf.variable_scope('learning_rate'):
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.polynomial_decay(
            params['initial_learning_rate'],
            global_step=global_step,
            decay_steps=params['decay_steps'],
            end_learning_rate=params['end_learning_rate'],
            name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops), tf.variable_scope('optimizer'):
        optimizer = tf.train.MomentumOptimizer(learning_rate,
                                               momentum=MOMENTUM,
                                               name='Momentum')
        grads_and_vars = optimizer.compute_gradients(total_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

    tf.summary.scalar('train_accuracy', accuracy[1])
    tf.summary.scalar('train_top5_accuracy', top5_accuracy[1])

    with tf.control_dependencies([train_op]), tf.name_scope('ema'):
        ema = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY,
                                                num_updates=global_step)
        train_op = ema.apply(tf.trainable_variables())

    if INIT_FLAG:
        INIT_FLAG=False
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op, training_hooks=[RestoreHook(init_fn)])
    else:
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)


def add_weight_decay(weight_decay):
    weights = [
        v for v in tf.trainable_variables()
        if 'weights' in v.name and 'depthwise_weights' not in v.name
    ]
    for w in weights:
        value = tf.multiply(weight_decay, tf.nn.l2_loss(w))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, value)


class RestoreMovingAverageHook(tf.train.SessionRunHook):
    def __init__(self, model_dir):
        super(RestoreMovingAverageHook, self).__init__()
        self.model_dir = model_dir

    def begin(self):
        ema = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY)
        variables_to_restore = ema.variables_to_restore()
        self.load_ema = tf.contrib.framework.assign_from_checkpoint_fn(
            tf.train.latest_checkpoint(self.model_dir), variables_to_restore)

    def after_create_session(self, sess, coord):
        tf.logging.info('Loading EMA weights...')
        self.load_ema(sess)

class RestoreHook(tf.train.SessionRunHook):
    def __init__(self, init_fn):
        self.init_fn = init_fn

    def after_create_session(self, session, coord=None):
        if session.run(tf.train.get_or_create_global_step()) == 0:
            self.init_fn(session)