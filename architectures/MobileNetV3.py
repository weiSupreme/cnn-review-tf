import tensorflow as tf
import tensorflow.contrib.slim as tfslim

mobilenetv3_large = {
    'kernel': [3, 3, 3, 5, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5],
    'expand':
    [16, 64, 72, 72, 120, 120, 240, 200, 184, 184, 480, 672, 672, 672, 960],
    'output':
    [16, 24, 24, 40, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160, 960],
    'SE': [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    'activation': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'stride': [1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1]
}
mobilenetv3_samll = {
    'kernel': [3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5],
    'expand': [16, 72, 88, 96, 240, 240, 120, 144, 288, 576, 576],
    'output': [16, 24, 24, 40, 40, 40, 48, 48, 96, 96, 96, 576],
    'SE': [1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'activation': [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    'stride': [2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1]
}
CONFIGURATIONS = {'0.5': mobilenetv3_samll, '1.0': mobilenetv3_large}

BATCH_NORM_MOMENTUM = 0.997
BATCH_NORM_EPSILON = 1e-36

expand_multiplier = 3  # scaled to 1/3


def MobileNetV3(x, is_training, num_classes=200, depth_multiplier='0.5'):
    configs = CONFIGURATIONS[depth_multiplier]
    num_bnecks = len(configs['kernel'])

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

    params = {'normalizer_fn': batch_norm}
    with tf.variable_scope('MobileNetV3'):
        with tfslim.arg_scope([tfslim.conv2d, depthwise_conv], **params):
            x = tfslim.conv2d(x,
                              num_outputs=16,
                              kernel_size=3,
                              stride=2,
                              activation_fn=Hswish,
                              scope='conv1')
            for unit in range(num_bnecks):
                with tf.variable_scope('bneck' + str(unit + 1)):
                    ratio = expand_multiplier
                    if unit == 0:
                        ratio = 1
                    if configs['stride'][unit] == 1:
                        x = basic_block(x,
                                        num_outputs=configs['output'][unit],
                                        expand=configs['expand'][unit] //
                                        ratio,
                                        kernel_size=configs['kernel'][unit],
                                        is_SE=configs['SE'][unit],
                                        is_Hswish=configs['activation'][unit])
                    else:
                        x = downsample(x,
                                       num_outputs=configs['output'][unit],
                                       expand=configs['expand'][unit] // ratio,
                                       kernel_size=configs['kernel'][unit],
                                       is_SE=configs['SE'][unit],
                                       is_Hswish=configs['activation'][unit])
            x = tfslim.conv2d(x,
                              num_outputs=configs['output'][num_bnecks],
                              kernel_size=1,
                              activation_fn=Hswish,
                              scope='conv' + str(2 + num_bnecks))
        with tf.variable_scope('global_pool'):
            x = tf.reduce_mean(x, axis=[1, 2])
            x = Hswish(x)
        x = tfslim.fully_connected(x,
                                   num_outputs=1280,
                                   activation_fn=Hswish,
                                   scope='fc')
        x = tfslim.fully_connected(x,
                                   num_outputs=num_classes,
                                   activation_fn=None,
                                   scope='classifier')
    return x


def basic_block(x,
                num_outputs,
                expand,
                kernel_size=3,
                stride=1,
                is_SE=0,
                is_Hswish=0):
    in_channels = x.shape[3].value
    activation = tf.nn.relu6 if not is_Hswish else Hswish
    with tf.variable_scope('residual'):
        y = tfslim.conv2d(x,
                          num_outputs=expand,
                          kernel_size=1,
                          activation_fn=activation,
                          scope='conv1x1_before')
        y = depthwise_conv(y, kernel=kernel_size, activation_fn=activation)
        y = tfslim.conv2d(y,
                          num_outputs=num_outputs,
                          kernel_size=1,
                          activation_fn=None,
                          scope='conv1x1_after')
        if is_SE:
            y = SE_block(y)
    if in_channels != num_outputs:
        x = tfslim.conv2d(x,
                          num_outputs=num_outputs,
                          kernel_size=1,
                          activation_fn=tf.nn.relu6,
                          scope='shortcut')
    y = y + x
    return y


def downsample(x,
               num_outputs,
               expand,
               kernel_size=3,
               stride=2,
               is_SE=0,
               is_Hswish=0):
    activation = tf.nn.relu6 if not is_Hswish else Hswish
    x = tfslim.conv2d(x,
                      num_outputs=expand,
                      kernel_size=1,
                      activation_fn=activation,
                      scope='conv1x1_before')
    x = depthwise_conv(x,
                       kernel=kernel_size,
                       stride=stride,
                       activation_fn=activation)
    x = tfslim.conv2d(x,
                      num_outputs=num_outputs,
                      kernel_size=1,
                      activation_fn=None,
                      scope='conv1x1_after')
    if is_SE:
        x = SE_block(x)
    return x


def Hswish(x):
    return x * (tf.nn.relu6(x + 3.0) / 6.0)


@tf.contrib.framework.add_arg_scope
def depthwise_conv(x,
                   kernel=3,
                   stride=1,
                   padding='SAME',
                   activation_fn=tf.nn.relu6,
                   normalizer_fn=None,
                   weights_initializer=tf.contrib.layers.xavier_initializer(),
                   data_format='NHWC',
                   scope='depthwise_conv'):

    with tf.variable_scope(scope):
        assert data_format == 'NHWC'
        in_channels = x.shape[3].value
        W = tf.get_variable('depthwise_weights',
                            [kernel, kernel, in_channels, 1],
                            dtype=tf.float32,
                            initializer=weights_initializer)
        x = tf.nn.depthwise_conv2d(x,
                                   W, [1, stride, stride, 1],
                                   padding,
                                   data_format='NHWC')
        x = normalizer_fn(
            x) if normalizer_fn is not None else x  # batch normalization
        x = activation_fn(
            x) if activation_fn is not None else x  # nonlinearity
        return x


def SE_block(x):
    in_channels = x.shape[3].value
    with tf.variable_scope('SE'):
        y = tf.reduce_mean(x, axis=[1, 2], name='global_pool')
        y = tfslim.fully_connected(y,
                                   num_outputs=in_channels // 16,
                                   activation_fn=tf.nn.relu6,
                                   scope='fc1')
        y = tfslim.fully_connected(y,
                                   num_outputs=in_channels,
                                   activation_fn=tf.nn.sigmoid,
                                   scope='fc2')
        y = tf.reshape(y, [-1, 1, 1, in_channels])
        x = x * y
    return x