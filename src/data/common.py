import numpy as np
from PIL import Image


def smooth(im, step=25, minv=0, maxv=255):
    im = np.asarray(im)
    ret = im.copy()
    for v in np.arange(minv, maxv, step):
        sel = (v <= im) & (im < v + step)
        if len(im[sel]) > 0:
            ret[sel] = np.mean(im[sel])
    # print(f'#unique values reduced from {len(np.unique(im))} to {len(np.unique(ret))}')
    return Image.fromarray(ret.clip(0, 255))
