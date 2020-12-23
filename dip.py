import os
from PIL import Image


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
    pre_name = make_filepath(args.content, tag='pre', ext_name='png')
    im = Image.open(args.content).convert('L').convert('RGB')
    im.save(pre_name)
    args.content = pre_name
    print(f'preprocess pencil {pre_name}')


def pre_ink(args):
    pre_name = make_filepath(args.content, tag='pre', ext_name='png')
    im = Image.open(args.content).convert('L').convert('RGB')
    im.save(pre_name)
    args.content = pre_name
    print(f'preprocess ink {pre_name}')


def match_color(args):
    from skimage.io import imread, imsave
    from skimage.exposure import match_histograms

    reference = imread(args.content)
    image = imread(args.style)

    matched = match_histograms(image, reference, multichannel=True)
    hm_name = make_filepath(args.style, tag='hm', ext_name='png')
    print(f'match color to {hm_name}')
    imsave(hm_name, matched)
    args.style = hm_name


def oil_handler(args):
    match_color(args)


def water_handler(args):
    match_color(args)


def pencil_handler(args):
    pre_pencil(args)


def ink_handler(args):
    pre_ink(args)


handler = { 'oil': oil_handler, 'water': water_handler, 'ink': ink_handler, 'pencil': pencil_handler}
