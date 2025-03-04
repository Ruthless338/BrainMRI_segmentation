def dice_coefficient(pred, target):
    smooth = 1e-5  # 防止分母为零
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice


def jaccard_index(pred, target):
    smooth = 1e-5
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


def precision_recall(pred, target):
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + fn + 1e-5)
    return precision, recall


def f1_score(pred, target):
    precision, recall = precision_recall(pred, target)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-5)
    return f1


def accuracy(pred, target):
    tp = (pred * target).sum()
    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-5)
    return acc


def evaluate_model(pred, target):
    dice = dice_coefficient(pred, target)
    iou = jaccard_index(pred, target)
    precision, recall = precision_recall(pred, target)
    f1 = f1_score(pred, target)
    acc = accuracy(pred, target)
    return {
        "Dice": dice.item(),
        "IoU": iou.item(),
        "Precision": precision.item(),
        "Recall": recall.item(),
        "F1 Score": f1.item(),
        "Accuracy": acc.item()
    }
