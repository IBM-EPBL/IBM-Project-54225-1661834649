import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch

batch_size = 32

def trainDataset():
    dataset = dset.ImageFolder(root="../input/100-bird-species/train",
                            transform=transforms.Compose([
                                transforms.RandomRotation(30),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomResizedCrop(224, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ]))
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4)
    return dataloader                               

def testDataset():
    dataset = dset.ImageFolder(root="../input/100-bird-species/test",
                            transform=transforms.Compose([
                                transforms.Resize([224, 224]),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                            ]))
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4)
    return dataloader
