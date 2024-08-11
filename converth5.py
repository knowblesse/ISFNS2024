import numpy as np
import h5py
from pathlib import Path
import sys
import tifffile as tf

def convert_h5(dataset_path):
    dataset_path = Path(dataset_path[0])
    image_path = sorted(list(dataset_path.glob('*.tif')))[0]
    #create in ctzyx
    img = tf.imread(image_path)
    img = np.expand_dims(np.expand_dims(img, axis=1), axis=0)
    empty_img = np.zeros_like(img)
    img = np.concatenate((empty_img, img), axis=0).astype(np.int16)

    h5_path = dataset_path / 'data.h5'
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('data', data=img)


if __name__ == '__main__':
    convert_h5(sys.argv[1:])


