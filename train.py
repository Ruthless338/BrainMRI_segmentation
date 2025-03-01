from torch.utils.data import DataLoader
from UNet import UNet
import torch
from myDatasets import myDataset
from diceLoss import DiceLoss
from eval import evaluate_model
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

# 超参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = '/root/autodl-tmp/UNet'
model = UNet(3, 2, True, 64).to(device)
lr = 0.05
max_iterations = 3
optimizer = torch.optim.SGD(model.parameters(), lr)
criterion = DiceLoss()  # diceLoss
state_dict = torch.load("/root/autodl-tmp/state_dict.pth")
model.load_state_dict(state_dict['model_state_dict'])
optimizer.load_state_dict(state_dict['optimizer_state_dict'])


if __name__ == '__main__':
    trainData = myDataset(root, True)
    n = trainData.__len__()
    train_loader = DataLoader(trainData, batch_size=8, shuffle=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # print(trainData.__len__())

    for epoch in range(max_iterations):
        model.train()
        total_loss = 0
        cnt = 0
        for batch in train_loader:
            MRI, mask = batch
            MRI, mask = MRI.to(device), mask.to(device)  # 确保数据移动到正确的设备
            optimizer.zero_grad()
            output = model(MRI)
            output = torch.sigmoid(output)  # 如果使用 DiceLoss，通常需要对输出进行 sigmoid 激活
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            cnt += 1
            print(f'Epoch {epoch + 1}, {cnt}-th, Loss: {loss:.4f}')
            # if cnt % 10 == 0:  # 每10个batch可视化一次
            #     for i in range(MRI.size(0)):  # 遍历批次中的每个样本
            #         original_img = MRI[i]  # 原图
            #         true_mask = mask[i]  # 标签图
            #         predicted_mask = output[i]  # 预测图
            #         plot_images(original_img, true_mask, predicted_mask)
        scheduler.step()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}')

    state_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': max_iterations
    }
    torch.save(state_dict, "state_dict.pth")

    testData = myDataset(root, False)
    test_loader = DataLoader(testData, batch_size=8, shuffle=False, collate_fn=myDataset.collate_fn)

    model.eval()
    results = []
    with torch.no_grad():
        for batch in test_loader:
            MRI, mask = batch
            MRI, mask = MRI.to(device), mask.to(device)
            output = model(MRI)
            pred = (torch.sigmoid(output) > 0.5).float()  # 二值化预测结果
            metrics = evaluate_model(pred, mask)
            results.append(metrics)

    # 打印平均指标
    avg_metrics = {k: sum([r[k] for r in results]) / len(results) for k in results[0].keys()}
    print("Average Test Metrics:", avg_metrics)