import tensorflow as tf
import numpy as np

# @staticmethod


def do_nothing(inputs):
    return inputs


def crop_and_resize(image):
    # A random box for cropping is generated per input image
    box_margin = 0.15  # tried 0.5, 0.33, 0.1 before
    random_box = tf.transpose(
        [tf.random_uniform(shape=[1], minval=0., maxval=box_margin, dtype=tf.float32, name='randombox_y1'),
         tf.random_uniform(shape=[1], minval=0., maxval=box_margin, dtype=tf.float32, name='randombox_x1'),
         tf.random_uniform(shape=[1], minval=1.-box_margin, maxval=1., dtype=tf.float32, name='randombox_y2'),
         tf.random_uniform(shape=[1], minval=1.-box_margin, maxval=1., dtype=tf.float32, name='randombox_x2')
         ]
    )

    image = tf.expand_dims(image, 0)  # tf.image.crop_and_resize() expects 4-D tensor. Thus, add an extra dimension
    image = tf.cond(
        tf.squeeze(tf.greater_equal(tf.random_uniform(shape=[1], minval=0., maxval=1.0, dtype=tf.float32,
                                                      name='coin_toss'),
                                    tf.constant(value=0.5, dtype=tf.float32, name='coin_threshold')
                                    )
                   ),
        true_fn=lambda: tf.image.crop_and_resize(image=image,
                                                 boxes=random_box,
                                                 box_ind=tf.range(0, image.get_shape().as_list()[0]),
                                                 crop_size=[image.get_shape().as_list()[1], image.get_shape().as_list()[2]],
                                                 method='bilinear',
                                                 extrapolation_value=0,
                                                 name='crop_and_resize'
                                                 ),
        false_fn=lambda: do_nothing(image)
    )
    image = tf.squeeze(image)  # Remove the extra dimension added for tf.image.crop_and_resize()

    return image


def data_augmentation(inputs):
    scope = 'data_aug'
    with tf.variable_scope(scope):
        #  ############  data augmentation on the fly  ###################
        #  PART 0: First, randomly crop images. Randomness has two folds:
        #  i) Coin toss: To crop or not to crop ii) Bounding box corners are randomly generated
        #  This part can considered as an extension to the sampling process.
        #  random crops from images resized to original shape (zooming effect)
        inputs = tf.map_fn(lambda img: crop_and_resize(img), inputs, dtype=tf.float32, name='random_crop_resize')

        # PART 1: Color transformations via brightness, hue, saturation, and contrast adjustments

        # random brightness adjustment with delta sampled from [-max_delta, max_delta]. must be in [0,1)
        # So far best/optimal without bleak images [0.15]
        inputs = tf.map_fn(lambda img: tf.image.random_brightness(img, max_delta=0.15), inputs,
                           dtype=tf.float32, name='random_brightness')

        # random hue adjustments with delta sampled from [-max_delta, max_delta]. max_delta must be in [0, 0.5].
        # So far best/optimal without bleak images [0.15]
        inputs = tf.map_fn(lambda img: tf.image.random_hue(img, max_delta=0.15), inputs, dtype=tf.float32,
                           name='random_hue')

        # random saturation adjustments: i) lower >= 0 and ii) lower < upper
        # So far best/optimal without bleak images [0.5, 1.5]
        inputs = tf.map_fn(lambda img: tf.image.random_saturation(img, lower=0.5, upper=1.5), inputs,
                           dtype=tf.float32, name='random_saturation')

        # random contrast adjustments: i) lower >= 0 and ii) lower < upper
        # So far best/optimal without bleak images [0.5, 2.5]
        inputs = tf.image.random_contrast(inputs, lower=0.5, upper=2.5)

        # make sure that pixel values are in [0., 1.]
        # inputs = tf.minimum(inputs, 1.0)
        # inputs = tf.maximum(inputs, 0.0)

        # rescale into [0,1] via min-max normalization (OUTPUT IMAGES SEEM BUGGY!!! DO NOT USE!)
        # input_min = tf.reduce_min(inputs, axis=[0], keepdims=True,  name='inputs_min')
        # input_max = tf.reduce_max(inputs, axis=[0], keepdims=True, name='inputs_min')
        #
        # new_max = tf.constant(value=1., dtype=tf.float32, name='new_max')
        # new_min = tf.constant(value=0., dtype=tf.float32, name='new_min')
        #
        # inputs = tf.add(new_min, tf.multiply(tf.divide(tf.subtract(inputs, input_min),
        #                                                tf.subtract(input_max, input_min)),
        #                                      tf.subtract(new_max, new_min)
        #                                      )
        #                 )

        # PART 2: Geometric transformations on images: Flip LR, Flip UD, Rotate

        # randomly mirror images horizontally
        inputs = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), inputs, dtype=tf.float32,
                           name='random_flip_lr')

        # randomly mirror images vertically
        inputs = tf.map_fn(lambda img: tf.image.random_flip_up_down(img), inputs, dtype=tf.float32,
                           name='random_flip_ud')

        # random translations
        inputs = tf.contrib.image.translate(inputs,
                                            translations=tf.random_uniform(shape=[tf.shape(inputs)[0], 2],
                                                                           minval=-25, maxval=25, dtype=tf.float32
                                                                           ),
                                            interpolation='NEAREST',
                                            name=None
                                            )

        # random rotations
        rotation_angle = 15

        tf_pi = tf.constant(np.pi, dtype=tf.float32, shape=None, name='PI_constant', verify_shape=False)
        tf_180 = tf.constant(180, dtype=tf.float32, shape=None, name='180_degrees', verify_shape=False)

        min_degree = tf.constant(-rotation_angle, dtype=tf.float32, shape=None,
                                 name='min_rotation_angle_degree', verify_shape=False)
        max_degree = tf.constant(rotation_angle, dtype=tf.float32, shape=None,
                                 name='max_rotation_angle_degree', verify_shape=False)

        min_radian = tf.divide(tf.multiply(min_degree, tf_pi, name='min_rotation_angle_radian_partial'), tf_180,
                               name='min_rotation_angle_radian')
        max_radian = tf.divide(tf.multiply(max_degree, tf_pi, name='max_rotation_angle_radian_partial'), tf_180,
                               name='max_rotation_angle_radian')

        inputs = tf.contrib.image.rotate(inputs,
                                         angles=tf.random_uniform(shape=[tf.shape(inputs)[0]],
                                                                  minval=min_radian, maxval=max_radian,
                                                                  dtype=tf.float32
                                                                  ),
                                         interpolation='NEAREST'
                                         )
    return inputs


def data_augmentation_wrapper(inputs, data_aug_prob=0.5):

    threshold = 1. - data_aug_prob

    print('Data augmentation probability : %g' % data_aug_prob)
    print('Biased-coin threshold : %g' % threshold)

    inputs = tf.cond(
        tf.squeeze(tf.greater_equal(tf.random_uniform(shape=[1], minval=0., maxval=1.0, dtype=tf.float32,
                                                      name='coin_toss'),
                                    tf.constant(value=threshold, dtype=tf.float32, name='data_augmentation_threshold')
                                    )
                   ),
        true_fn=lambda: data_augmentation(inputs),
        false_fn=lambda: do_nothing(inputs)
    )

    return inputs
