import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import numpy
from torchvision import models

dataset = dset.ImageFolder(root="sample",
                        transform=transforms.Compose([
                            transforms.Resize([224, 224]),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        ]))
dataloader = torch.utils.data.DataLoader(dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=0)


model = models.resnet50()
model.load_state_dict(torch.load('checkpoint/epoche-5', map_location='cpu'))
model.eval()


output = model(list(dataloader)[0][0])
_, predicted = torch.max(output, 1)

if predicted[0].numpy() == 1:
    print("normal")
else:
    print("diseased")