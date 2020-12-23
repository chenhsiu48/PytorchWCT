#!/usr/bin/env python3

import torchvision.utils as vutils
from Loader import DatasetOne
from util import *
import time
import dip
import glob


def styleTransfer(wct, targets, contentImg, styleImg, imname, gamma, delta, outf, transform_method):
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

        decoder_input = (gamma * (delta * CsIlast + (1 - delta) * CsIorig) + (1 - gamma) * eIorig)
        decoder_input = decoder_input.unsqueeze(0).to(next(wct.parameters()).device)

        decoder = wct.decoders[target]
        current_result = decoder(decoder_input)

    # save_image has this wired design to pad images with 4 pixels at default.
    vutils.save_image(current_result.cpu().float(), os.path.join(outf, imname))
    return current_result


def exec_transfer(args):
    dataset = DatasetOne(args.content, args.style, args.fineSize)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    wct = WCT(args)
    if args.cuda:
        wct.cuda(args.gpu)

    avgTime = 0
    with torch.no_grad():
        for i, (contentImg, styleImg, imname) in enumerate(loader):
            if (args.cuda):
                contentImg = contentImg.cuda(args.gpu)
                styleImg = styleImg.cuda(args.gpu)
            imname = imname[0]
            print(f'Transferring {imname}')
            if (args.cuda):
                contentImg = contentImg.cuda(args.gpu)
                styleImg = styleImg.cuda(args.gpu)
            start_time = time.time()
            # WCT Style Transfer
            targets = [f'relu{t}_1' for t in args.targets]
            styleTransfer(wct, targets, contentImg, styleImg, imname,
                          args.gamma, args.delta, args.outf, args.transform_method)
            end_time = time.time()
            print('Elapsed time is: %f' % (end_time - start_time))
            avgTime += (end_time - start_time)
    print('Processed %d images. Averaged time is %f' % ((i + 1), avgTime / (i + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--contentPath', default='images/content', help='path to train')
    parser.add_argument('--stylePath', default='style', help='path to train')
    parser.add_argument('--content', default=None, help='')
    parser.add_argument('--style', default=None, help='')
    parser.add_argument('--effect', default=None, help='water, ink, oil, pencil')
    parser.add_argument('--workers', default=2, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--encoder', default='models/vgg19_normalized.pth.tar', help='Path to the VGG conv1_1')
    parser.add_argument('--decoder5', default='models/vgg19_normalized_decoder5.pth.tar', help='Path to the decoder5')
    parser.add_argument('--decoder4', default='models/vgg19_normalized_decoder4.pth.tar', help='Path to the decoder4')
    parser.add_argument('--decoder3', default='models/vgg19_normalized_decoder3.pth.tar', help='Path to the decoder3')
    parser.add_argument('--decoder2', default='models/vgg19_normalized_decoder2.pth.tar', help='Path to the decoder2')
    parser.add_argument('--decoder1', default='models/vgg19_normalized_decoder1.pth.tar', help='Path to the decoder1')
    parser.add_argument('--cuda', default=True, help='enables cuda')
    parser.add_argument('--transform-method', choices=['original', 'closed-form'], default='original',
                        help=('How to whiten and color the features. "original" for the formulation of Li et al. ( https://arxiv.org/abs/1705.08086 )  '
                              'or "closed-form" for method of Lu et al. ( https://arxiv.org/abs/1906.00668 '))
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--fineSize', type=int, default=0, help='resize image to fineSize x fineSize,leave it to 0 if not resize')
    parser.add_argument('--outf', default='output/', help='folder to output images')
    parser.add_argument('--targets', default=[5, 4, 3, 2, 1], nargs='+', help='which layers to stylize at. Order matters!')
    parser.add_argument('--gamma', type=float, default=0.8, help='hyperparameter to blend original content feature and colorized features. See Wynen et al. 2018 eq. (3)')
    parser.add_argument('--delta', type=float, default=0.9, help='hyperparameter to blend wct features from current input and original input. See Wynen et al. 2018 eq. (3)')
    parser.add_argument('--gpu', type=int, default=0, help="which gpu to run on.  default is 0")

    args = parser.parse_args()

    dip.ensure_dir(args.outf)

    if args.content is None:
        print(f'missing --content')
    elif args.effect is not None:
        if args.effect not in dip.handler.keys():
            print(f'effect {args.effect} not supported')
            exit(1)

        styles = glob.glob(dip.join_path(args.stylePath, f'{args.effect}*.jpg'))
        for s in styles:
            org_content = args.content
            args.style = s
            dip.handler[args.effect](args)
            exec_transfer(args)
            args.content = org_content

    elif args.style is None:
        print(f'missing --style')
    else:
        exec_transfer(args)
