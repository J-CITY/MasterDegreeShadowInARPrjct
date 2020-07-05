import torch.utils.data
from data.base_data_loader import BaseDataLoader


class CustomDatasetDataLoader(BaseDataLoader):
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self
    
    def get_number_of_patches(self):
        return self.dataset.get_number_of_patches()
    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data

def CreateDataset():
    dataset = None
    from data.image_folder import ImageFolder
    dataset = ImageFolder('../../dataset/sbu')
    return dataset


def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt)
    return data_loader
