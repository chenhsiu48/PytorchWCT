import os
from PIL import Image
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def join_path(*dirs):
    if len(dirs) == 0:
        return ''
    path = dirs[0]
    for d in dirs[1:]:
        path = os.path.join(path, d)
    return path


def make_filepath(fpath, dir_name=None, ext_name=None, tag=None):
    if dir_name is None:
        dir_name = os.path.dirname(fpath)
        if dir_name == '':
            dir_name = '.'
    fname = os.path.basename(fpath)
    base, ext = os.path.splitext(fname)
    if ext_name is None:
        ext_name = ext
    elif ext_name != '' and ext_name[0] != '.':
        ext_name = '.' + ext_name
    name = base
    if tag == '':
        name = name.split('-')[0]
    elif tag is not None:
        name = '%s-%s' % (name, tag)
    if ext_name != '':
        name = '%s%s' % (name, ext_name)
    return join_path(dir_name, name)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def pre_pencil(args):
    pre_name = make_filepath(args.content, tag='pre_pencil', ext_name='png')
    print(f'preprocess pencil {pre_name}')
    im = Image.open(args.content).convert('L')

    image = np.array(im)
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, sal_map) = saliency.computeSaliency(image)

    sal_map = gaussian_filter(sal_map, sigma=im.width / 8 / 2)
    edges = cv2.Canny(image, 50, 150)
    edges = gaussian_filter(edges, sigma=2)

    sal_map /= np.max(sal_map)
    image = image + image * (1 - sal_map) - edges
    image = np.clip(image, 0, 255)

    im = Image.fromarray(image)
    im = im.convert('RGB')

    im.save(pre_name)
    args.content = pre_name

    pre_name = make_filepath(args.style, tag='edit', ext_name='png')
    match_color(pre_name, args.content, args.style)
    args.style = pre_name


def pre_ink(args):
    pre_name = make_filepath(args.content, tag='pre_ink', ext_name='png')
    print(f'preprocess ink {pre_name}')
    im = Image.open(args.content).convert('L')

    image = np.array(im)
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, sal_map) = saliency.computeSaliency(image)

    sal_map = gaussian_filter(sal_map, sigma=im.width / 16 / 2)
    edges = cv2.Canny(image, 50, 150)
    edges = gaussian_filter(edges, sigma=2)

    sal_map /= np.max(sal_map)
    image = image + image * (1 - sal_map) - edges
    image = np.clip(image, 0, 255)

    im = Image.fromarray(image)
    im = im.convert('RGB')

    im.save(pre_name)
    args.content = pre_name

    #pre_name = make_filepath(args.content, ext_name='png')
    #match_color(pre_name, args.style, args.content)
    #args.content = pre_name


def match_color(pre_name, ref_img, target_img):
    from skimage.io import imread, imsave
    from skimage.exposure import match_histograms

    reference = imread(ref_img)
    image = imread(target_img)

    matched = match_histograms(image, reference, multichannel=True)
    print(f'match color to {pre_name}')
    imsave(pre_name, matched)


def oil_handler(args):
    pre_name = make_filepath(args.style, tag='edit', ext_name='png')
    match_color(pre_name, args.content, args.style)
    args.style = pre_name


def water_handler(args):
    pre_name = make_filepath(args.style, tag='edit', ext_name='png')
    match_color(pre_name, args.content, args.style)
    args.style = pre_name


def pencil_handler(args):
    pre_pencil(args)


def ink_handler(args):
    pre_ink(args)


handler = { 'oil': oil_handler, 'water': water_handler, 'ink': ink_handler, 'pencil': pencil_handler}
