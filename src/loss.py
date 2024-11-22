from torch.nn.functional import  binary_cross_entropy_with_logits, sigmoid

def dice_loss_with_logits(predictions, targets, smooth=1e-6):
    return dice_loss(sigmoid(predictions), targets, smooth=smooth)

def dice_loss(predictions, targets, smooth=1e-6):
    total = (predictions * predictions).sum() + (targets * targets).sum()
    intersection = (predictions * targets).sum()                            
    dice = (2.0 * intersection + smooth) / (total + smooth)  
    return 1.0 - dice

def dice_bce_loss_with_logits(predictions, targets, smooth=1e-6):
    return 0.5 * binary_cross_entropy_with_logits(predictions, targets) + 0.5 * dice_loss_with_logits(predictions, targets, smooth=smooth)

if __name__ == '__main__':
    print(1)