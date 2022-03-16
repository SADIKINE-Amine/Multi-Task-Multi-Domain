import math
import torch
from torch.utils.data import RandomSampler
from ipdb import set_trace
import os, sys

sys.path.append('/home/sadikine/code/Multi-Task-Multi-Domain')

class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([len(cur_dataset.data) for cur_dataset in dataset.datasets])
    # def __len__(self):
    #     return int(self.largest_dataset_size/self.batch_size)*len(self.dataset.datasets)

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size #* self.number_of_datasets
        samples_to_grab = int(self.batch_size/self.number_of_datasets)

        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.largest_dataset_size * self.number_of_datasets

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)
        return iter(final_samples_list)

from torch.utils.data.dataset import ConcatDataset




if __name__ == "__main__":
    from datasets import IRCAD_Dataset as Dataset
    from    arguments               import get_args
    from    transforms              import Deftransforms

    datalist1 = [
                 os.path.join("01-VE.nii.gz"),
                 os.path.join("02-VE.nii.gz"),
                 os.path.join("03-VE.nii.gz"),
                 os.path.join("04-VE.nii.gz"),
                 os.path.join("05-VE.nii.gz"),
                 os.path.join("06-VE.nii.gz"),
                 os.path.join("07-VE.nii.gz"),
                 os.path.join("08-VE.nii.gz"),
                 os.path.join("09-VE.nii.gz"),
                 os.path.join("10-VE.nii.gz"),
                 os.path.join("11-VE.nii.gz"),
                 os.path.join("12-VE.nii.gz"),
                 os.path.join("13-VE.nii.gz"),
                 os.path.join("14-VE.nii.gz"),
                 os.path.join("15-VE.nii.gz"),
                 os.path.join("16-VE.nii.gz"),
                 os.path.join("17-VE.nii.gz"),
                 os.path.join("18-VE.nii.gz"),
                 os.path.join("19-VE.nii.gz"),
                 os.path.join("20-VE.nii.gz")
                ]

    datalist2 = [
                 os.path.join("001-VE.nii.gz"),
                 os.path.join("002-VE.nii.gz"),
                 os.path.join("003-VE.nii.gz"),
                 os.path.join("004-VE.nii.gz"),
                 os.path.join("005-VE.nii.gz"),
                 os.path.join("006-VE.nii.gz"),
                 os.path.join("007-VE.nii.gz"),
                 os.path.join("008-VE.nii.gz"),
                 os.path.join("009-VE.nii.gz"),
                 os.path.join("010-VE.nii.gz"),
                 os.path.join("011-VE.nii.gz"),
                 os.path.join("012-VE.nii.gz"),
                 os.path.join("013-VE.nii.gz"),
                 os.path.join("014-VE.nii.gz"),
                 os.path.join("015-VE.nii.gz"),
                 os.path.join("016-VE.nii.gz"),
                 os.path.join("017-VE.nii.gz"),
                 os.path.join("018-VE.nii.gz"),
                 os.path.join("019-VE.nii.gz"),
                 os.path.join("020-VE.nii.gz"),
                 os.path.join("022-VE.nii.gz"),
                 os.path.join("023-VE.nii.gz"),
                 os.path.join("025-VE.nii.gz"),
                 os.path.join("027-VE.nii.gz"),
                 os.path.join("028-VE.nii.gz"),
                 os.path.join("030-VE.nii.gz"),
                ]

    class MyFirstDataset(torch.utils.data.Dataset):
        def __init__(self):
            # dummy dataset
            self.samples = datalist1

        def __getitem__(self, index):
            # change this to your samples fetching logic
            return self.samples[index]

        def __len__(self):
            # change this to return number of samples in your dataset
            return len(self.samples)


    class MySecondDataset(torch.utils.data.Dataset):
        def __init__(self):
            # dummy dataset
            self.samples =  datalist2

        def __getitem__(self, index):
            # change this to your samples fetching logic
            return self.samples[index]

        def __len__(self):
            # change this to return number of samples in your dataset
            return len(self.samples)


    first_dataset = MyFirstDataset()
    second_dataset = MySecondDataset()
    concat_dataset = ConcatDataset([first_dataset, second_dataset])


    args    = get_args()
    batch_size = 4

    train_transforms, val_transforms = Deftransforms(Dataset=args.DatasetName)
    train_ds        = Dataset(dataset_dir=args.DataPath, section="training", anatomy=args.anatomy_target, transform=train_transforms)
    concat_dataset = ConcatDataset([train_ds, train_ds])

    # dataloader with BalancedBatchSchedulerSampler
    dataloader = torch.utils.data.DataLoader(dataset=concat_dataset,sampler=BatchSchedulerSampler(dataset=concat_dataset,batch_size=batch_size), batch_size=batch_size, shuffle=False)

    set_trace()
    # itr = iter(dataloader)
    # data= next(itr)
    # data["image"]
    for inputs in dataloader:
        print(inputs)