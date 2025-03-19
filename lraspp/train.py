import torch
import torch.nn as nn
from model import lraspp_mobilenetv3_large 
from myDatasets import myDataset
from torch.utils.data import DataLoader
from diceLoss import DiceLoss
from eval import evaluate_model



lr=0.001
epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = lraspp_mobilenetv3_large(num_classes=1, pretrain_backbone=False).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
root = 'C:/Users/陈毅彪/source/repos/py/BrainMRI_segmentaion/MRI'
# root = 'autodl-tmp/BrainMRI_segmentaion/MRI'
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
criterion = DiceLoss()



if __name__ == '__main__':
    train_dataset = myDataset(root, train=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i,(MRI,mask) in enumerate(train_loader):
            MRI = MRI.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            output = model(MRI)
            output = output["out"]
            loss = criterion(output,MRI)
            loss.backward()
            optimizer.step()
        print('epoch:{} loss:{}'.format(epoch, total_loss/len(train_loader)))
        scheduler.step()

    state_dict = {
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'epoch':epochs,
    }
    torch.save(state_dict, 'state_dict.pth')
    model.eval()
    results = []
    test_dataset = myDataset(root, train=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    with torch.no_grad():
        for i,(MRI,mask) in enumerate(test_loader):
            MRI = MRI.to(device)
            mask = mask.to(device)
            output = model(MRI)
            output = output["out"]
            output = torch.Sigmoid(output)
            output = (output>0.5).float()
            results.append(evaluate_model(output, mask))
        
    avg_metrics = {k: sum([r[k] for r in results]) / len(results) for k in results[0].keys()}
    print("Average Test metrics: ", avg_metrics)