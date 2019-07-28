from PIL import Image
import argparse
import numpy as np
import sys
import os
from os import walk

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--indir', action='store', help='image path')
parser.add_argument('-o', '--outdir', action='store', help='output dir', default='JPEGImages')
parser.add_argument('-w', '--width', action='store', help='width', default='513')
args = parser.parse_args()



flist = []
for (dirpath, dirnames, filenames) in walk(args.indir):
    flist.extend(filenames)
    break

print("Images:", flist)


if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)


basewidth = int(args.width)

for fname in flist:
    fpath = args.indir + '/' + fname
    im = Image.open(fpath)
    print(fpath, im.format, im.mode, im.size)

    wpercent = (basewidth / float(im.size[0]))
    hsize = int((float(im.size[1]) * float(wpercent)))
    img = im.resize((basewidth, hsize))

    print(fpath, img.format, img.mode, img.size)

    img.save(args.outdir + '/' + fname )
