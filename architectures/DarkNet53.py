import tensorflow as tf
import tensorflow.contrib.slim as tfslim

repeat = [1, 2, 8, 8, 4]
filters = [64, 128, 256, 512, 1024]

BATCH_NORM_MOMENTUM = 0.997
BATCH_NORM_EPSILON = 1e-36


def DarkNet53(x, is_training, num_classes=200, depth_multiplier='0.5'):
    def batch_norm(x):
        x = tf.layers.batch_normalization(x,
                                          axis=3,
                                          center=True,
                                          scale=True,
                                          training=is_training,
                                          momentum=BATCH_NORM_MOMENTUM,
                                          epsilon=BATCH_NORM_EPSILON,
                                          fused=True,
                                          name='batch_norm')
        return x

    params = {'normalizer_fn': batch_norm, 'activation_fn': tf.nn.leaky_relu}
    with tf.variable_scope('DarkNet53'):
        with tfslim.arg_scope([tfslim.conv2d], **params):
            x = tfslim.conv2d(x, num_outputs=32, kernel_size=3, scope='conv1')
            for stage in range(len(repeat)):
                with tf.variable_scope('stage' + str(stage + 2)):
                    x = tfslim.conv2d(x,
                                      num_outputs=filters[stage],
                                      kernel_size=3,
                                      stride=2,
                                      activation_fn=tf.nn.leaky_relu,
                                      scope='downsample')
                    for unit in range(repeat[stage]):
                        with tf.variable_scope('uint' + str(unit + 1)):
                            x = block(x, filters[stage])
        x = tf.reduce_mean(x, axis=[1, 2])
        x = tfslim.fully_connected(x,
                                   num_classes,
                                   activation_fn=None,
                                   scope='classifier')
    return x


def block(x, num_outputs):
    x = tfslim.conv2d(x,
                      num_outputs=num_outputs // 2,
                      kernel_size=1,
                      scope='conv1x1')
    x = tfslim.conv2d(x,
                      num_outputs=num_outputs,
                      kernel_size=3,
                      stride=1,
                      scope='conv3x3')
    with tf.variable_scope('residual'):
        y = tfslim.conv2d(x,
                          num_outputs=num_outputs,
                          kernel_size=1,
                          scope='conv1x1')
        y = tfslim.conv2d(y,
                          num_outputs=num_outputs,
                          kernel_size=3,
                          scope='conv3x3')
        x = y + x
        return x