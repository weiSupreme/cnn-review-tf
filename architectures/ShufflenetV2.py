import tensorflow as tf
import tensorflow.contrib.slim as tfslim

CHNNELS = {
    '0.5': [24, 48, None, None, 1024],
    '1.0': [24, 116, None, None, 1024],
    '1.5': [24, 176, None, None, 1024],
    '2.0': [24, 244, None, None, 2048]
}
REPEAT = [1, 4, 8, 4, 1]

BATCH_NORM_MOMENTUM = 0.997
BATCH_NORM_EPSILON = 1e-3


def ShuffleNetV2(x, is_training, num_classes=200, depth_multiplier='0.5'):
    out_channels = CHNNELS[depth_multiplier]

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

    with tf.variable_scope('ShuffleNetV2'):
        params = {'normalizer_fn': batch_norm}

        with tfslim.arg_scope([tfslim.conv2d, tfslim.separable_conv2d],
                              **params):
            x = tfslim.conv2d(x,
                              out_channels[0],
                              kernel_size=3,
                              stride=2,
                              scope='conv1')
            x = tfslim.max_pool2d(x,
                                  kernel_size=3,
                                  stride=2,
                                  scope='max_pool1')
            for st in [2, 3, 4]:
                with tf.variable_scope('stage' + str(st)):
                    x, y = downsample(x, out_channels[st - 1])
                    for u in range(2, REPEAT[st - 1] + 1):
                        with tf.variable_scope('unit' + str(u)):
                            x, y = concat_shuffle_split(x, y)
                            y = basic_unit(y)
                    x = tf.concat([x, y], axis=3)
            x = tfslim.conv2d(x, out_channels[4], kernel_size=1, scope='conv5')
        x = tf.reduce_mean(x, axis=[1, 2])
        x = tfslim.fully_connected(x,
                                   num_classes,
                                   activation_fn=None,
                                   scope='classifier')
    return x


def downsample(x, out_channels=None):
    in_channels = x.shape[3].value
    out_channels = in_channels * 2 if not out_channels else out_channels
    with tf.variable_scope('downsample'):
        with tf.variable_scope('first_branch'):
            y = tfslim.conv2d(x, in_channels, kernel_size=1, scope='conv1x1')
            y = tfslim.separable_conv2d(y,
                                        out_channels // 2,
                                        kernel_size=3,
                                        depth_multiplier=1,
                                        stride=2,
                                        scope='separable_conv')
        with tf.variable_scope('second_branch'):
            x = tfslim.separable_conv2d(x,
                                        out_channels // 2,
                                        kernel_size=3,
                                        depth_multiplier=1,
                                        stride=2,
                                        scope='separable_conv')
    return x, y


def concat_shuffle_split(x, y):
    with tf.name_scope('concat_shuffle_split'):
        shape = tf.shape(x)
        batch_size = shape[0]
        height, width = shape[1], shape[2]
        depth = x.shape[3]

        z = tf.stack([x, y],
                     axis=3)  # shape [batch_size, height, width, 2, depth]
        z = tf.transpose(z, [0, 1, 2, 4, 3])
        z = tf.reshape(z, [batch_size, height, width, 2 * depth])
        x, y = tf.split(z, num_or_size_splits=2, axis=3)
        return x, y


def basic_unit(x):
    in_channels = x.shape[3]
    with tf.variable_scope('basic_unit'):
        x = tfslim.conv2d(x, in_channels, kernel_size=1, scope='conv1x1')
        x = tfslim.separable_conv2d(x,
                                    in_channels,
                                    kernel_size=3,
                                    depth_multiplier=1,
                                    scope='separable_conv')
    return x
