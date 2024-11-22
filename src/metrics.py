import torch
from torchmetrics import Metric

class MIoU(Metric):
    def __init__(self, n_classes=2):
        super().__init__()
        self.n_classes = n_classes
        self.add_state("intersection", default=torch.zeros(n_classes), dist_reduce_fx="sum")
        self.add_state("union", default=torch.zeros(n_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = torch.round(preds)
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        for cls in range(self.n_classes):
            self.intersection[cls] += torch.logical_and(preds == cls, target == cls).sum()
            self.union[cls] += torch.logical_or(preds == cls, target == cls).sum()

    def compute(self) -> torch.Tensor:
        return (self.intersection / (self.union + 1e-10)).mean()
    

if __name__ == '__main__':
    n_classes=2
    miou=MIoU(2)
    preds = torch.tensor([
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0]
    ])
    target = torch.tensor([
        [0, 1, 0],
        [0, 1, 1],
        [1, 1, 0]
    ])

    # 更新指标
    miou.update(preds, target)

    # 计算并输出结果
    result = miou.compute()
    print(f"mIoU: {result:.4f}")