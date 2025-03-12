import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import u2net_full, u2net_lite
from myDatasets import myDataset
from eval import evaluate_model
import os
from diceLoss import DiceLoss


root = 'C:/Users/陈毅彪/source/repos/py/BrainMRI_segmentaion/MRI'
# root = '/root/auto-tmp/BrainMRI_segmentation/MRI'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = u2net_full(1).to(device)
criterion = DiceLoss()
lr = 0.001
epochs = 50
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

if __name__ == '__main__':
    train_dataset = myDataset(root, train=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=myDataset.collate_fn)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            MRI, mask = batch
            MRI = MRI.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            output = model(MRI)
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print('Epoch: {}, Loss: {:.4f}'.format(epoch, total_loss / len(train_loader)))

    state_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epochs
    }
    torch.save(state_dict, "state_dict.pth")
    results = []
    model.eval()
    testData = myDataset(root, False)
    test_loader = DataLoader(testData, batch_size=16, shuffle=False, collate_fn=myDataset.collate_fn)
    with torch.no_grad():
        for batch in test_loader:
            MRI, mask = batch
            MRI = MRI.to(device)
            mask = mask.to(device)
            output = model(MRI)
            output = torch.sigmoid(output)
            output = (output > 0.5).float()
            results.append(evaluate_model(output, mask))

    avg_metrics = {k: sum([r[k] for r in results]) / len(results) for k in results[0].keys()}
    print("Average Test metrics: ", avg_metrics)