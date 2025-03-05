import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms  import ToTensor,RandomHorizontalFlip,RandomRotation,ColorJitter



# train共88个文件夹，test共23个文件夹，每个文件有20+组原图像+mask图像
# root/MRI/train
# root/MRI/test
class myDataset(Dataset):
    # root是当前项目文件夹的路径
    def __init__(self, root: str, train: bool):
        super(Dataset, self).__init__()
        self.flag = "train" if train else "test"
        data_root = os.path.join(root, self.flag)
        # 检查路径是否存在
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transformer = transforms.Compose([
            transforms.Resize((256, 256)),
            RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=15),
            ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor()
        ])
        # 递归遍历每个子文件夹，存储所有图片的路径
        self.MRI = []
        self.mask = []
        for folder_name in os.listdir(data_root):
            folder_path = os.path.join(data_root, folder_name)
            if os.path.isdir(folder_path):
                # MRI和掩膜图像的文件名相似，但掩膜图像的文件名以 _mask.tif 结尾
                MRI_names = [i for i in os.listdir(folder_path) if i.endswith(".tif") and "_mask" not in i]
                for MRI_name in MRI_names:
                    MRI_path = os.path.join(folder_path, MRI_name)
                    mask_name = MRI_name.replace(".tif", "_mask.tif")
                    mask_path = os.path.join(folder_path, mask_name)
                    if os.path.exists(mask_path):  # 检查掩膜图像路径是否存在
                        mask = Image.open(mask_path)
                        mask_tensor = ToTensor()(mask)
                        if mask_tensor.sum() > 0:
                            self.MRI.append(MRI_path)
                            self.mask.append(mask_path)
                        # self.MRI.append(MRI_path)
                        # self.mask.append(mask_path)
                    else:
                        print(f"Warning: file {mask_path} does not exists. Skipping {MRI_name}")
        # 检查掩膜图像路径是否都存在
        if len(self.MRI) != len(self.mask):
            raise ValueError("The number of MRI images and mask images do not match.")

    # 返回索引为id的核磁共振图像和标签掩膜图像
    def __getitem__(self, id):
        MRI = Image.open(self.MRI[id])
        mask = Image.open(self.mask[id])
        MRI = self.transformer(MRI)
        mask = self.transformer(mask)  # 转换为张量后，掩膜图像的值范围是 [0, 1]
        # mask = torch.squeeze(mask).type(torch.long)
        mask = (mask > 0.5).float()  # 将掩膜图像二值化
        return MRI, mask

    # 返回总的图片数
    def __len__(self):
        return len(self.MRI)

    # batch是(image,target)这样的列表
    # collate_fn把images和targets打包成两个批次张量
    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


# 将多个不同大小的图片/张量 合成一个更高维度的数据结构（批次张量）
def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs




from torch.utils.data import DataLoader

root = ''


if __name__ == '__main__':
    trainData = myDataset(root, True)
    n = trainData.__len__()
    train_loader = DataLoader(trainData, batch_size=8, shuffle=True, collate_fn=myDataset.collate_fn)
    print(trainData.__len__())

    for (MRI, mask) in train_loader:
        print(mask.max())
