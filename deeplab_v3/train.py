import torch
from torch.utils.data import DataLoader
from model import deeplabv3_resnet101
from myDatasets import myDataset


root = ''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
