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


def get_saliency_map(image, sigma=24, drop_pct=0.1):
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, sal_map) = saliency.computeSaliency(image)

    s = sorted(list(sal_map.reshape(-1)))
    th = s[int(len(s) * drop_pct)]
    sal_map[sal_map <= th] = 0

    sal_map = gaussian_filter(sal_map, sigma=sigma)
    sal_map /= np.max(sal_map)
    return sal_map


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def pre_pencil(args):
    pre_name = make_filepath(args.content, tag='pre_pencil', ext_name='png')
    print(f'preprocess pencil {pre_name}')
    im = Image.open(args.content).convert('L')

    image = np.array(im)
    sal_map = get_saliency_map(image, sigma=25, drop_pct=0.1)
    sal_map = (sal_map * 255).astype(np.uint8)
    sal_map = adjust_gamma(sal_map, 2).astype(np.float) / 255.0

    edges = cv2.Canny(image, 50, 150)
    edges = gaussian_filter(edges, sigma=max(1, min(image.shape) // 200))

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
    print(f'preprocess pencil {pre_name}')
    im = Image.open(args.content)

    image = np.array(im.convert('L'))
    sal_map = get_saliency_map(image, sigma=50, drop_pct=0.2)
    sal_map = np.stack((sal_map, sal_map, sal_map), axis=2)

    sal_map = (sal_map * 255).astype(np.uint8)
    sal_map = adjust_gamma(sal_map, 1.5).astype(np.float) / 255.0

    white = np.full_like(im, 255)

    im = (1 - sal_map) * white + sal_map *  im

    im = Image.fromarray(im.astype(np.uint8))

    im.save(pre_name)
    args.content = pre_name


def match_color(pre_name, ref_img, target_img):
    from skimage.io import imread, imsave
    from skimage.exposure import match_histograms

    reference = imread(ref_img)
    image = imread(target_img)

    matched = match_histograms(image, reference, multichannel=True)
    print(f'match color to {pre_name}')
    imsave(pre_name, matched)


def oil_handler(args):
    pre_name = make_filepath(args.content, tag='pre_oil', ext_name='png')
    print(f'preprocess oil {pre_name}')
    im = Image.open(args.content)

    image = np.array(im)
    hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    s = adjust_gamma(s, 1.5)
    v = adjust_gamma(v, 0.9)
    hsv = np.stack((h, s, v), axis=2)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    im = Image.fromarray(image)

    im.save(pre_name)
    args.content = pre_name

    pre_name = make_filepath(args.style, tag='edit', ext_name='png')
    match_color(pre_name, args.content, args.style)
    args.style = pre_name


def water_handler(args):
    pre_name = make_filepath(args.content, tag='pre_water', ext_name='png')
    print(f'preprocess water {pre_name}')
    im = Image.open(args.content)

    image = np.array(im)
    hsv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    s = adjust_gamma(s, 0.75)
    v = adjust_gamma(v, 1.1)
    hsv = np.stack((h, s, v), axis=2)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    im = Image.fromarray(image)

    im.save(pre_name)
    args.content = pre_name

    pre_name = make_filepath(args.style, tag='edit', ext_name='png')
    match_color(pre_name, args.content, args.style)
    args.style = pre_name


def pencil_handler(args):
    pre_pencil(args)


def ink_handler(args):
    pre_ink(args)


handler = { 'oil': oil_handler, 'water': water_handler, 'ink': ink_handler, 'pencil': pencil_handler}
