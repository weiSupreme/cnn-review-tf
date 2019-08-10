import tensorflow as tf
import tensorflow.contrib.slim as tfslim


def VGG16(x, is_training, num_classes=200, depth_multiplier='0.5'):
    with tf.variable_scope('VGG16'):
        #feature extractor

        #224*224
        x = tfslim.conv2d(x, num_outputs=64, kernel_size=3, scope='conv1_1')
        x = tfslim.conv2d(x, num_outputs=64, kernel_size=3, scope='conv1_2')
        x = tfslim.max_pool2d(x, kernel_size=2, scope='max_pool1')

        #112*112
        x = tfslim.conv2d(x, num_outputs=128, kernel_size=3, scope='conv2_1')
        x = tfslim.conv2d(x, num_outputs=128, kernel_size=3, scope='conv2_2')
        x = tfslim.max_pool2d(x, kernel_size=2, scope='max_pool2')

        #conv3,conv4,conv5
        outputs = [256, 512, 512]
        for i in [3, 4, 5]:
            for j in [1, 2, 3]:
                x = tfslim.conv2d(x,
                                  num_outputs=outputs[i - 3],
                                  kernel_size=3,
                                  scope='conv' + str(i) + '_' + str(j))
            x=tfslim.max_pool2d(x,kernel_size=2,scope='max_pool'+str(i))

        x=tfslim.flatten(x,scope='flatten')
        #fc layers
        for i in [1,2]:
            x=tfslim.fully_connected(x,num_outputs=4096,scope='fc'+str(i))
        
        #classifier
        x=tfslim.fully_connected(x,num_outputs=num_classes,activation_fn=None,scope='classifier')

    return x