import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


class DatasetParams:
    train_path = "dataset/train"
    valid_path = "dataset/valid"
    classes = ['otto_mann', 'mayor_quimby', 'carl_carlson', 'sideshow_bob', 'patty_bouvier', 'troy_mcclure', 'gil',
               'selma_bouvier', 'waylon_smithers', 'agnes_skinner', 'marge_simpson', 'moe_szyslak', 'cletus_spuckler', 'principal_skinner',
               'edna_krabappel', 'rainier_wolfcastle', 'martin_prince', 'charles_montgomery_burns', 'lisa_simpson', 'lionel_hutz', 'lenny_leonard',
               'sideshow_mel', 'ralph_wiggum', 'professor_john_frink', 'milhouse_van_houten', 'bart_simpson', 'barney_gumble', 'ned_flanders',
               'snake_jailbird', 'kent_brockman', 'comic_book_guy', 'chief_wiggum', 'miss_hoover', 'krusty_the_clown', 'apu_nahasapeemapetilon',
               'nelson_muntz', 'maggie_simpson', 'fat_tony', 'homer_simpson', 'abraham_grampa_simpson', 'disco_stu', 'groundskeeper_willie']
    batch_size = 64
    num_workers = 8
    mean = [0.4623, 0.4075, 0.3524]
    std = [0.2131, 0.1922, 0.2214]
    
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        "valid": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }


class CustomDataset(Dataset):
    def __init__(self, root_dir, classes, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = classes

        self.images = []
        self.labels = []

        # filling pathes to images and labels
        for i, image_class in enumerate(classes):
            class_path = os.path.join(root_dir, image_class)
            for image_name in os.listdir(class_path):
                self.images.append(os.path.join(class_path, image_name))
                self.labels.append(i)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        image_obj = Image.open(image_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform is not None:
            image_obj = self.transform(image_obj)

        return image_obj, label


def calculate_mean_std(dataset, batch_size, num_workers):
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0

    with tqdm(dataloader, desc="Calculating progress") as pbar:
        for images in pbar:
            batch_size = images[0].size(0)
            images = images[0].view(batch_size, images[0].size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_samples += batch_size
            
            pbar.set_postfix({"samples": total_samples})

    mean /= total_samples
    std /= total_samples

    return mean, std


def get_mean_std():
    valid = CustomDataset(DatasetParams.valid_path,
                          classes=DatasetParams.classes)
    full = CustomDataset(DatasetParams.train_path,
                         classes=DatasetParams.classes,
                         transform=transforms.Compose([transforms.Resize([256, 256]),
                                                       transforms.ToTensor()]))
    full.images += valid.images
    full.labels += valid.labels
    return calculate_mean_std(dataset=full,
                              batch_size=DatasetParams.batch_size,
                              num_workers=DatasetParams.num_workers)

