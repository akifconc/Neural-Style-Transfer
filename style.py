from PIL import Image
import tensorflow as tf
import numpy as np
import vgg
from sys import stderr
import time


CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

try:
    reduce
except NameError:
    from functools import reduce
'''
This function is used to style the image, it takes parameter like style weight, content weight, style image, content image
and generates our final image. Optimization is also performed in this function.
'''

def stylize(network, initial, content, styles):
   
    print_iterations = None
    checkpoint_iterations = None
    shape = (1,) + content.shape
    style_shapes = (1,) + styles.shape
    content_features = {}
    style_features = [{} for _ in styles]

    vgg_weights, vgg_mean_pixel = vgg.load_net(network)

    layer_weight = 1.0
    style_layers_weights = {}
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] = layer_weight

    # the weight of style layer is normalized
    layer_weights_sum = 0
    for style_layer in STYLE_LAYERS:
        layer_weights_sum += style_layers_weights[style_layer]
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] /= layer_weights_sum

    # extract the content feature as mentioned in the original paper
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=shape)
        net = vgg.net_preloaded(vgg_weights, image, 'max')
        content_pre = np.array([vgg.preprocess(content, vgg_mean_pixel)])
        for layer in CONTENT_LAYERS:
            content_features[layer] = net[layer].eval(feed_dict={image: content_pre})

    # extract the style feature as mentioned in the original paper
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=style_shapes)
        net = vgg.net_preloaded(vgg_weights, image, 'max')
        style_pre = np.array([vgg.preprocess(styles, vgg_mean_pixel)])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={image: style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[0][layer] = gram

    initial_content_noise_coeff = 1.0 - 1

    # generate our final image
    with tf.Graph().as_default():
        if initial is None:
            noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
            initial = tf.random_normal(shape) * 0.256
        else:
            initial = np.array([vgg.preprocess(initial, vgg_mean_pixel)])
            initial = initial.astype('float32')
            noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
            initial = (initial) * initial_content_noise_coeff + (tf.random_normal(shape) * 0.256) * (1.0 - initial_content_noise_coeff)
        image = tf.Variable(initial)
        net = vgg.net_preloaded(vgg_weights, image, 'max')

        # calculate the content cost function
        content_layers_weights = {}
        content_layers_weights['relu4_2'] = 1
        content_layers_weights['relu5_2'] = 1.0 - 1

        content_loss = 0
        content_losses = []
        for content_layer in CONTENT_LAYERS:
            content_losses.append(content_layers_weights[content_layer] * 5e0 * (2 * tf.nn.l2_loss(
                    net[content_layer] - content_features[content_layer]) /
                    content_features[content_layer].size))
        content_loss += reduce(tf.add, content_losses)

        # calculate the style cost function using gram matrices
        style_loss = 0
        style_losses = []
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            _, height, width, number = map(lambda i: i.value, layer.get_shape())
            size = height * width * number
            feats = tf.reshape(layer, (-1, number))
            gram = tf.matmul(tf.transpose(feats), feats) / size
            style_gram = style_features[0][style_layer]
            style_losses.append(style_layers_weights[style_layer] * 2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)
        style_loss += 5e2 * reduce(tf.add, style_losses)

        # overall loss
        loss = content_loss + style_loss

        # optimizer setup
        train_step = tf.train.AdamOptimizer(1e1, 0.9, 0.999, 1e-08).minimize(loss)

        def print_progress():
            stderr.write('  content loss: %g\n' % content_loss.eval())
            stderr.write('    style loss: %g\n' % style_loss.eval())
            stderr.write('    total loss: %g\n' % loss.eval())

        # optimization
        best_loss = float('inf')
        best = None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            stderr.write('Optimization started...\n')
            iteration_times = []
            start = time.time()
            for i in range(1000):
                iteration_start = time.time()
                if i > 0:
                    elapsed = time.time() - start
                    # take average of last couple steps to get time per iteration
                    remaining = np.mean(iteration_times[-10:]) * (1000 - i)
                    stderr.write('Iteration %4d/%4d (%s elapsed, %s remaining)\n' % (
                        i + 1,
                        1000,
                        hms(elapsed),
                        hms(remaining)
                    ))
                else:
                    stderr.write('Iteration %4d/%4d\n' % (i + 1, 1000))
                train_step.run()

                last_step = (i == 1000 - 1)
                if last_step or (print_iterations and i % print_iterations == 0):
                    print_progress()

                if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                    this_loss = loss.eval()
                    if this_loss < best_loss:
                        best_loss = this_loss
                        best = image.eval()

                    img_out = vgg.unprocess(best.reshape(shape[1:]), vgg_mean_pixel)

                    yield (
                        (None if last_step else i),
                        img_out
                    )

                iteration_end = time.time()
                iteration_times.append(iteration_end - iteration_start)


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb

def hms(seconds):
    seconds = int(seconds)
    hours = (seconds // (60 * 60))
    minutes = (seconds // 60) % 60
    seconds = seconds % 60
    if hours > 0:
        return '%d hr %d min' % (hours, minutes)
    elif minutes > 0:
        return '%d min %d sec' % (minutes, seconds)
    else:
        return '%d sec' % seconds
