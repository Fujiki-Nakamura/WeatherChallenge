import datetime as dt
from pathlib import Path

import hickle as hkl
import numpy as np
import pandas as pd
from PIL import Image


factor = 4
height, width = int(672 / factor), int(512 / factor)
channel = 1
OUTPUT_D = Path('../inputs/hkl/')


def create_test_data(csvpath, target=False):
    df = pd.read_csv(csvpath)
    if target:
        start_list = list(df.loc[:, 'Inference_24hr_Start'].values)
        end_list = list(df.loc[:, 'Inference_24hr_End'].values)
    else:
        start_list = list(df.loc[:, 'OpenData_96hr_Start'].values)
        end_list = list(df.loc[:, 'OpenData_96hr_End'].values)
    assert len(start_list) == len(end_list) == len(df)

    split = 'train' if 'validation' in csvpath else 'test'
    data_d = Path(f'../inputs/{split}/sat/')
    seq_list = []
    for i in range(len(df)):
        start_str = start_list[i]
        end_str = end_list[i]
        start_dt = dt.datetime.strptime(start_str, '%Y/%m/%d %H:%M')
        end_dt = dt.datetime.strptime(end_str, '%Y/%m/%d %H:%M')

        seq = []
        current_dt = start_dt
        while current_dt <= end_dt:
            y, m, d = current_dt.year, current_dt.month, current_dt.day
            h = current_dt.hour
            dname = f'{y:0>2}-{m:0>2}-{d:0>2}'
            fname = f'{dname}-{h:0>2}-00.fv.png'
            impath = data_d/dname/fname
            if impath.is_file():
                im = Image.open(impath)
                im = im.resize((width, height))
                im = np.asarray(im)[:, :, np.newaxis]
            else:
                im = np.zeros((height, width, channel))
            seq.append(im)
            current_dt += dt.timedelta(hours=1)

        stacked = np.stack(seq, axis=0)
        seq_list.append(stacked)

    X = np.stack(seq_list, axis=0)
    assert len(df) == len(X)

    data_type = 'y' if target else 'X'
    split_type = 'valid' if 'valid' in csvpath else 'test'
    path = OUTPUT_D/f'{data_type}_{split_type}_{height}x{width}.hkl'
    hkl.dump(X, path.as_posix())
    print(f'Dumped at {path}')


if __name__ == '__main__':
    create_test_data('../inputs/validation.csv')
    create_test_data('../inputs/validation.csv', target=True)
    create_test_data('../inputs/inference_terms.csv')
