import torch
from myDatasets import myDataset
from torch.utils.data import DataLoader
from plt import plot_images
import os
from model import FCN_resnet50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FCN_resnet50(True, 1, False).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
current_dir = os.path.dirname(__file__)
root = "c:/Users/陈毅彪/source/repos/py/BrainMRI_segmentaion/MRI"
pth = os.path.join(current_dir, 'state_dict_1_40')
state_dict = torch.load(pth, map_location=torch.device('cpu'))
optimizer.load_state_dict(state_dict['optimizer_state_dict'])
model.load_state_dict(state_dict['model_state_dict'])

if __name__ == '__main__':
    test_data = myDataset(root, False)
    test_loader = DataLoader(test_data, 8, False, collate_fn=myDataset.collate_fn)

    model.eval()
    results = []
    with torch.no_grad():
        for batch in test_loader:
            MRI, mask = batch
            MRI, mask = MRI.to(device), mask.to(device)
            output = model(MRI)
            pred = (torch.sigmoid(output["out"]) > 0.5).float()  # 二值化预测结果
            for i in range(MRI.size(0)):
                original_img = MRI[i]  # 原图
                true_mask = mask[i]  # 标签图
                predicted_mask = pred[i]  # 预测图
                plot_images(original_img, true_mask, predicted_mask)
