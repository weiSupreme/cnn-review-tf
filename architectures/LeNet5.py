import tensorflow as tf
import tensorflow.contrib.slim as tfslim


def LeNet5(x, is_training, num_classes=200, depth_multiplier='0.5'):
    with tf.variable_scope('LeNet5'):
        #feature extractor
        x = tfslim.conv2d(x,
                          num_outputs=6,
                          kernel_size=5,
                          padding='VALID',
                          scope='conv1')
        x = tfslim.max_pool2d(x, kernel_size=2, scope='max_pool1')
        x = tfslim.conv2d(x,
                          num_outputs=16,
                          kernel_size=5,
                          padding='VALID',
                          scope='conv2')
        x = tfslim.max_pool2d(x, kernel_size=2, scope='max_pool2')

        #flatten
        x=tfslim.flatten(x,scope='flatten')

        #classifier
        x = tfslim.fully_connected(x, num_outputs=120, scope='fc1')
        logits = tfslim.fully_connected(x,
                                        num_classes,
                                        activation_fn=None,
                                        scope='classifier')
    return x