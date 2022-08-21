from utils import *
from glob import glob
from sklearn.model_selection import KFold

"""
должно все выглядеть примерно так:
list_of_mask = [...]
list_of_slides = [...]
list_of_datasets = [Dataset(slide, mask for slide, mask in zip(lists)]
kf = KFold(n_splits=len(list_of_slides))
for train_idx, test_idx in kf.split(list_of_datasets):
    train = ConcatDataset(list_of_datasets[train_idx]) #тут надо сделать это красиво, как тебе захочется
    train = list_of_datasets[test_idx]

"""