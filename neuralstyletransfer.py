from PIL import Image
import os
import numpy as np
import scipy.misc
from style import stylize
import math
from argparse import ArgumentParser



VGG_PATH = 'imagenet-vgg-verydeep-19.mat'

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--contentimage',
            dest='content', help='content image',
            metavar='CONTENT', required=True)
    parser.add_argument('--styleimage',
            dest='styles',help='style images',
            metavar='STYLE', required=True)
    parser.add_argument('--outputimage',
            dest='output', help='output path',
            metavar='OUTPUT', required=True)
    parser.add_argument('--network',
            dest='network', help='path to network parameters (default %(default)s)',
            metavar='VGG_PATH', default=VGG_PATH)
    return parser

#this function initiates the whole process, takes the arguments provided by the user and start the styling process.
def main():
    parser = build_parser()
    options = parser.parse_args()
    style_scale=1.0

    if not os.path.isfile(options.network):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % options.network)

    content_image = imread(options.content)
    style_images = imread(options.styles)

    target_shape = content_image.shape

    initial = content_image


    try:
        imsave(options.output, np.zeros((500, 500, 3)))
    except:
        raise IOError('%s is not writable or does not have a valid file extension for an image file' % options.output)

    for iteration, image in stylize(
        network=options.network,
        initial=initial,
        content=content_image,
        styles=style_images,
    ):
        output_file = None
        combined_rgb = image
        output_file = options.output
        if output_file:
            imsave(output_file, combined_rgb)


def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        img = np.dstack((img,img,img))
    elif img.shape[2] == 4:
        img = img[:,:,:3]
    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)

if __name__ == '__main__':
    main()
