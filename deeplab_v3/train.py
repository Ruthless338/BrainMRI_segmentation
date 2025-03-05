import torch
import os
from torch.utils.data import DataLoader
from model import deeplabv3_resnet101
from myDatasets import myDataset
from diceLoss import DiceLoss


root = ''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = deeplabv3_resnet101(True, 1, False).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 50
criterion = DiceLoss()
current_dir = os.path.dirname(__file__)


if __name__ == '__main__':
    train_dataset = myDataset(root, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=train_dataset.collate_fn)
    for epoch in range(epochs):
        for i, (image, mask) in enumerate(train_loader):
            image = image.to(device)
            mask = mask.to(device)
            output= model(image)
            optimizer.zero_grad()
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()