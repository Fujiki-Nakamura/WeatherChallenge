import datetime as dt
from pathlib import Path

import hickle as hkl
import numpy as np
from PIL import Image


factor = 4
height, width = int(672 / factor), int(512 / factor)
channel = 1


def dump_hkl(start, end, datetime_format='%Y-%m-%d %H:%M'):
    start_dt = dt.datetime.strptime(start, datetime_format)
    end_dt = dt.datetime.strptime(end, datetime_format)
    assert start_dt.year == end_dt.year
    year = start_dt.year

    logf = open(f'../inputs/missing_{year}.list', 'a')
    inputs_d = Path('../inputs')
    data_d = inputs_d/'test/sat/' if year == 2018 else inputs_d/'train/sat/'
    output_d = inputs_d/'hkl/'

    im_list = []
    source_list = []
    current_dt = start_dt
    while current_dt <= end_dt:
        y, m, d, h = current_dt.year, current_dt.month, current_dt.day, current_dt.hour
        dname = f'{y:0>2}-{m:0>2}-{d:0>2}'
        fname = f'{dname}-{h:0>2}-00.fv.png'
        impath = (data_d/dname/fname)
        if impath.is_file():
            im = Image.open(impath).convert('L')
            im = im.resize((width, height))
            im = np.asarray(im)[:, :, np.newaxis]
        else:
            im = np.zeros((height, width, channel))
            print(f'File not found: {fname}', file=logf)
        im_list.append(im)
        source_list.append(f'year')
        current_dt = current_dt + dt.timedelta(hours=1)

    X = np.stack(im_list, axis=0)
    path = output_d/f'X_{year}_{height}x{width}.hkl'
    hkl.dump(X, path.as_posix())
    print(f'Dumped at {path}')
    path = output_d/f'source_{year}_{height}x{width}.hkl'
    hkl.dump(source_list, path.as_posix())
    print(f'Dumped at {path}')

    logf.close()


if __name__ == '__main__':
    # dump_hkl('2016-01-01 00:00', '2016-12-31 23:00')
    # dump_hkl('2017-01-01 00:00', '2017-12-31 23:00')
    dump_hkl('2018-01-01 00:00', '2018-12-31 23:00')
