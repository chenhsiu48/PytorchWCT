#!/usr/bin/env python3

import torchvision.utils as vutils
from Loader import DatasetOne
from util import *
import time
import re
import dip
import torch.nn as nn

def styleTransfer(wct, targets, contentImg, styleImg, imname, gamma, delta, transform_method, sal_map):
    current_result = contentImg
    eIorigs = [f.cpu().squeeze(0) for f in wct.encoder(contentImg, targets)]
    eIss = [f.cpu().squeeze(0) for f in wct.encoder(styleImg, targets)]

    for i, (target, eIorig, eIs) in enumerate(zip(targets, eIorigs, eIss)):
        print(f'    stylizing at {target}')

        if i == 0:
            eIlast = eIorig
        else:
            eIlast = wct.encoder(current_result, target).cpu().squeeze(0)

        CsIlast = wct.transform(eIlast, eIs, transform_method).float()
        CsIorig = wct.transform(eIorig, eIs, transform_method).float()

        if sal_map is not None:
            w = nn.functional.interpolate(sal_map, size=(CsIlast.shape[1], CsIlast.shape[2])).squeeze(0)
            w = 0.2 * w / torch.max(w)
            gP = gamma - w
            dP = delta - w / 2
        else:
            gP = gamma
            dP = delta

        decoder_input = (gP * (dP * CsIlast + (1 - dP) * CsIorig) + (1 - gP) * eIorig)
        decoder_input = decoder_input.unsqueeze(0).to(next(wct.parameters()).device)

        decoder = wct.decoders[target]
        current_result = decoder(decoder_input)

    # save_image has this wired design to pad images with 4 pixels at default.
    vutils.save_image(current_result.cpu().float(), imname)
    return current_result


def exec_transfer(args, sal_map):
    dataset = DatasetOne(args.content, args.style, args.fineSize)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    sal_map = torch.FloatTensor(sal_map).unsqueeze(0).unsqueeze(0)

    avgTime = 0
    with torch.no_grad():
        for i, (contentImg, styleImg, _) in enumerate(loader):
            if (args.cuda):
                contentImg = contentImg.cuda(args.gpu)
                styleImg = styleImg.cuda(args.gpu)
            imname = args.out_name
            print(f'Transferring {imname}')
            if (args.cuda):
                contentImg = contentImg.cuda(args.gpu)
                styleImg = styleImg.cuda(args.gpu)
            start_time = time.time()
            # WCT Style Transfer
            targets = [f'relu{t}_1' for t in args.targets]
            if args.no_saliency:
                styleTransfer(args.wct, targets, contentImg, styleImg, imname, args.gamma, args.delta, args.transform_method, None)
            else:
                styleTransfer(args.wct, targets, contentImg, styleImg, imname, args.gamma, args.delta, args.transform_method, sal_map)
            end_time = time.time()
            print('Elapsed time is: %f' % (end_time - start_time))
            avgTime += (end_time - start_time)
            im_transfer = Image.open(args.out_name)
            if args.small_content:
                print(f'resize the output to original size')
                im_transfer.resize((im_transfer.width // 2, im_transfer.height // 2)).save(args.out_name)
    print('Processed %d images. Averaged time is %f' % ((i + 1), avgTime / (i + 1)))
    return im_transfer


def handle_effect(args):
    db_name = dip.make_filepath(args.content, dir_name=args.outf, tag=f'debug-{args.effect}', ext_name='png')
    args.out_name = dip.make_filepath(args.content, dir_name=args.outf, tag=f'{args.e_tag}', ext_name='png')

    im_raw = Image.open(args.content)
    args.small_content = max(im_raw.size) < 500
    if args.small_content:
        name_2x = dip.make_filepath(args.content, tag='2x', ext_name='png')
        print(f'input image too small, double the size to {name_2x}')
        im_2x = im_raw.resize((im_raw.width * 2, im_raw.height * 2))
        im_2x.save(name_2x)
        args.cleanup.append(name_2x)
        args.content = name_2x

    (im, sal_map, im_edit, im_style, style_edit) = dip.handler[args.effect](args)
    im_transfer = exec_transfer(args, sal_map)

    if args.debug:
        im = im.resize(im_transfer.size)

        im_debug = Image.new('RGB', (im.width * 3, im.height * 2))
        im_debug.paste(im, (0, 0, im.width, im.height))
        im_debug.paste(Image.fromarray(sal_map * 255).resize(im_transfer.size).convert('RGB'), (im.width, 0, im.width * 2, im.height))
        im_debug.paste(im_edit.resize(im_transfer.size), (im.width * 2, 0, im.width * 3, im.height))
        im_debug.paste(im_style.resize(im_transfer.size), (0, im.height, im.width, im.height * 2))
        im_debug.paste(style_edit.resize(im_transfer.size), (im.width, im.height, im.width * 2, im.height * 2))
        im_debug.paste(im_transfer, (im.width * 2, im.height, im.width * 3, im.height * 2))

        print(f'save debug image {db_name}')
        im_debug.save(db_name)

        comp_name = dip.make_filepath(args.content, dir_name=args.outf, tag=f'comp-{args.effect}', ext_name='png')
        im_comp = Image.new('RGB', (im.width * 2, im.height))
        im_comp.paste(im, (0, 0, im.width, im.height))
        im_comp.paste(im_transfer, (im.width, 0, im.width * 2, im.height))
        if args.small_content:
            im_comp = im_comp.resize((im_comp.width // 2, im_comp.height // 2))
        print(f'save compare image {comp_name}')
        im_comp.save(comp_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('image', nargs = '*', help = 'images')
    parser.add_argument('--content', default=None, help='')
    parser.add_argument('--style', default=None, help='')
    parser.add_argument('--effect', choices=dip.handler.keys(), default=None, help='artistic style transfer effect')
    parser.add_argument('--cuda', default=True, help='enables cuda')
    parser.add_argument('--gpu', type=int, default=0, help="which gpu to run on.  default is 0")
    parser.add_argument('--transform-method', choices=['original', 'closed-form'], default='original',
                        help=('How to whiten and color the features. "original" for the formulation of Li et al. ( https://arxiv.org/abs/1705.08086 )  '
                              'or "closed-form" for method of Lu et al. ( https://arxiv.org/abs/1906.00668 '))
    parser.add_argument('--fineSize', type=int, default=0, help='resize image to fineSize x fineSize, leave it to 0 if not resize')
    parser.add_argument('--outf', default='output/', help='folder to output images')
    parser.add_argument('--targets', default=[5, 4, 3, 2, 1], nargs='+', help='which layers to stylize at. Order matters!')
    parser.add_argument('--gamma', type=float, default=0.9, help='hyperparameter to blend original content feature and colorized features. See Wynen et al. 2018 eq. (3)')
    parser.add_argument('--delta', type=float, default=0.95, help='hyperparameter to blend wct features from current input and original input. See Wynen et al. 2018 eq. (3)')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--no_saliency', action='store_true', help='no saliency')

    args = parser.parse_args()
    args.encoder = 'models/vgg19_normalized.pth.tar'
    args.decoder5 = 'models/vgg19_normalized_decoder5.pth.tar'
    args.decoder4 = 'models/vgg19_normalized_decoder4.pth.tar'
    args.decoder3 = 'models/vgg19_normalized_decoder3.pth.tar'
    args.decoder2 = 'models/vgg19_normalized_decoder2.pth.tar'
    args.decoder1 = 'models/vgg19_normalized_decoder1.pth.tar'
    args.cleanup = []

    dip.ensure_dir(args.outf)

    if args.content is None and len(args.image) == 0:
        print(f'missing --content or no image provided')
    elif args.style is not None:
        if args.effect is None:
            args.effect = re.sub('\d+', '', os.path.basename(args.style).split('.')[0])
        args.e_tag = os.path.basename(args.style).split('.')[0]

        args.wct = WCT(args)
        if args.cuda:
            args.wct.cuda(args.gpu)

        if args.content is not None:
            handle_effect(args)
        else:
            for fn in args.image:
                org_style = args.style
                args.content = fn
                handle_effect(args)
                args.style = org_style
    elif args.style is None:
        print(f'missing --style')

    if args.debug is False:
        dip.rm_files(args.cleanup)
