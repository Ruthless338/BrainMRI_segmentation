import matplotlib.pyplot as plt
import torch
'''
绘制图像
'''
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
    print(predicted_mask.permute(1, 2, 0).shape)
    # plt.imshow(torch.argmax(predicted_mask.permute(1, 2, 0), axis=-1).cpu().numpy())
    plt.imshow(predicted_mask.permute(1, 2, 0).cpu().numpy())
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.show()