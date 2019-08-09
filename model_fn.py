import tensorflow as tf
from architectures.LeNet5 import LeNet5 as cnn

MOMENTUM = 0.9
USE_NESTEROV = True
MOVING_AVERAGE_DECAY = 0.995
RESIZE_SIZE = 28


def model_fn(features, labels, mode, params):
    x = features['images']
    if mode == tf.estimator.ModeKeys.PREDICT:
        with tf.name_scope('image_preprocess'):
            x = tf.image.resize_images(x,
                                       [RESIZE_SIZE, RESIZE_SIZE],
                                       method=0)
            x = (1.0 / 255.0) * tf.to_float(x)
            x = tf.reshape(x, shape=[1, RESIZE_SIZE, RESIZE_SIZE, -1])
    else:
        with tf.name_scope('image_preprocess'):
            x = tf.reshape(x, shape=[-1, RESIZE_SIZE, RESIZE_SIZE, 1])
    with tf.name_scope('standardize_input'):
       x = (2.0 * x) - 1.0

    is_training = mode == tf.estimator.ModeKeys.TRAIN
    logits = cnn(x,
                 is_training,
                 num_classes=params['num_classes'],
                 depth_multiplier=params['depth_multiplier'])
    predicted_class = tf.argmax(logits, 1, output_type=tf.int32)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_class[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits, axis=1),
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    with tf.variable_scope('cross_entropy'):
        #compute loss
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                      logits=logits)
        #loss = tf.reduce_mean(loss, axis=0)

    #compute accuracy
    acc = tf.metrics.accuracy(labels=labels,
                              predictions=predicted_class,
                              name='accuracy')
    metrics = {'accuracy': acc}
    tf.summary.scalar('accuracy', acc[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode,
                                          loss=loss,
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

    with tf.variable_scope('optimizer'):
        optimizer = tf.train.MomentumOptimizer(learning_rate,
                                               momentum=MOMENTUM,
                                               name='Momentum')
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

    with tf.control_dependencies([train_op]), tf.name_scope('ema'):
        ema = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY,
                                                num_updates=global_step)
        train_op = ema.apply(tf.trainable_variables())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


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