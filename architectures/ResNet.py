import tensorflow as tf
import tensorflow.contrib.slim as tfslim

_scale={'0.5':'50','1.0':'101','1.5':'152'}
_repeat={'50':[3,4,6,3],'101':[3,4,23,3],'152':[3,8,36,3]}
_feaature_map_size=[64,128,256,512]

def ResNet(x, is_training, num_classes=200, depth_multiplier='0.5'):
    scale=_scale[depth_multiplier]
    repeat=_repeat[scale]
    with tf.variable_scope('ResNet'+scale):
        x=tfslim.conv2d(x,64,kernel_size=7,stride=2,scope='conv1')
        x=tfslim.max_pool2d(x,kernel_size=3,scope='max_pool1')
        for i in [2,3,4,5]:
            with tf.variable_scope('stage'+str(i)):
                for j in range(1,repeat[i-2]):
                    with tf.variable_scope('unit'+str(j)):
                        x=bottleneck(x,_feaature_map_size[i-2])
                with tf.variable_scope('downsample'):
                    x=bottleneck(x,_feaature_map_size[i-2],2)
        
        x=tf.reduce_mean(x,axis=[1,2])
        x=tfslim.fully_connected(x,num_classes,activation_fn=None,scope='classifier')
        return x


def bottleneck(x,num_channels,stride=1):
    with tf.variable_scope('bottleneck'):
        y=tfslim.conv2d(x,num_channels,kernel_size=1,scope='conv1x1_before')
        y=tfslim.conv2d(y,num_channels,kernel_size=3,stride=stride,scope='conv3x3')
        y=tfslim.conv2d(y,num_channels*4,kernel_size=1,activation_fn=None,scope='conv1x1_after')
        in_channel = tfslim.utils.last_dimension(x.get_shape(), min_rank=4)
        if in_channel!=num_channels*4 or stride>1:
            x=tfslim.conv2d(x,num_channels*4,kernel_size=1,stride=stride,scope='shortcut')
        y=y+x
    return y