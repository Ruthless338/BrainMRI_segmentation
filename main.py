from torch.utils.data import DataLoader, Dataset
from UNet import UNet
import torch
from myDatasets import myDataset
from diceLoss import DiceLoss

# 超参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = ''
model = UNet(3, 2, True, 64)
lr = 0.05
max_iterations = 10
optimizer = torch.optim.SGD(model.parameters(), lr)
criterion = DiceLoss()  # diceLoss

if __name__ == '__main__':

    trainData = myDataset(root, True)
    n = trainData.__len__()
    train_loader = DataLoader(trainData, batch_size=4, shuffle=True)
    for _ in range(max_iterations):
        model.train()
        loss = 0
        for i in range(n):
            MRI, mask = trainData.get(i)
            optimizer.zero_grad()
            output = model.forward(MRI)
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()
            print(f'Iteration {_ + 1}, Loss: {loss.item():.4f}')

    model.eval()
    testData = myDataset(root, False)
    n = testData.__len__()
    with torch.no_grad():
        for i in range(n):
            MRI, mask = testData.get(i)
            output = model(MRI)
            loss = criterion(output, mask)
            print(f'i-th test image loss:{loss.item():.4f}')
