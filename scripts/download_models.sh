#!/bin/bash
{
mkdir -p checkpoints/pretrained-models
prefix='checkpoints/pretrained-models'

wget https://www.dropbox.com/s/pxl3cba6yp78eqd/static-obstacle-detection.pth.tar?dl=0 -O $prefix/static-obstacle-detection.pth.tar

wget https://www.dropbox.com/s/vvv1olmr2pai8li/static-relative-depth.pth.tar?dl=0 -O $prefix/static-relative-depth.pth.tar

wget https://www.dropbox.com/s/bnveghppa6ni0az/motion-obstacle-detection.pth.tar?dl=0 -O $prefix/motion-obstacle-detection.pth.tar

wget https://www.dropbox.com/s/f3sgck397vwxpv8/motion-relative-depth.pth.tar?dl=0 -O $prefix/motion-relative-depth.pth.tar

wget https://www.dropbox.com/s/satqr3feervivc2/motion-avorder.pth.tar?dl=0 -O $prefix/motion-avorder.pth.tar

}