import torch
from UNet import UNet
from myDatasets import myDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def plot_images(original_img, true_mask, predicted_mask):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_img.permute(1, 2, 0).cpu().numpy())
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(true_mask.permute(1, 2, 0).cpu().numpy())
    plt.title('True Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(torch.argmax(predicted_mask.permute(1, 2, 0), axis=-1).cpu().numpy(), cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.show()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(3, 2, True).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
root = ''
pth = 'state_dict.pth'
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
            pred = (torch.sigmoid(output) > 0.5).float()  # 二值化预测结果
            for i in range(MRI.size(0)):
                original_img = MRI[i]  # 原图
                true_mask = mask[i]  # 标签图
                predicted_mask = output[i]  # 预测图
                plot_images(original_img, true_mask, predicted_mask)