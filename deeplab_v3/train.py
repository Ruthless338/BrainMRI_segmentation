import torch
import os
from torch.utils.data import DataLoader
from model import DeepLabV3_ResNet50
from myDatasets import myDataset
from diceLoss import DiceLoss
from eval import evaluate_model


root = 'C:/Users/陈毅彪/source/repos/py/BrainMRI_segmentaion/MRI'
# root = 'autodl-tmp/BrainMRI_segmentaion/MRI'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
aux = True
num_classes = 1
model = DeepLabV3_ResNet50(aux, num_classes, False).to(device)
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 50
criterion = DiceLoss()
current_dir = os.path.dirname(__file__)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)



if __name__ == '__main__':
    train_dataset = myDataset(root, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=train_dataset.collate_fn)
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for i, (MRI, mask) in enumerate(train_loader):
            MRI = MRI.to(device)
            mask = mask.to(device)
            output= model(MRI)
            main_out = output["out"]
            loss = criterion(main_out, mask)
            if aux:
                aux_out = output["aux"]
                loss += 0.5 * criterion(aux_out, mask)
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f'Epoch {epoch + 1}, Loss: {total_loss/len(train_dataset):.4f}')

    results = []
    torch.save(model.state_dict(), os.path.join(current_dir, 'state_dict.pth'))
    model.eval()
    test_dataset = myDataset(root, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=test_dataset.collate_fn)
    with torch.no_grad():
        for i, (MRI, mask) in enumerate(test_loader):
            MRI = MRI.to(device)
            mask = mask.to(device)
            output = model(MRI)
            output = torch.sigmoid(output)
            output = (output > 0.5).float()
            metrics = evaluate_model(output, mask)
            results.append(metrics)
    
    avg_metrics = {k: sum([r[k] for r in results]) / len(results) for k in results[0].keys()}
    print("Average Test metrics: ", avg_metrics)
        
            