from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


class FashionMNIST:

    def __init__(self, batch_size: int = 32, val_ratio: float = 0.2):
        """
        Initializes the Fashion MNIST Dataset.

        Args:
            batch_size: int
                Number of Batches
            val_ratio: float
                Ratio of the validation data.

        """
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])

        train_full = datasets.FashionMNIST(
            root="./data", 
            train=True, 
            download=True, 
            transform=transform
        )
    
        val_size = int(len(train_full) * val_ratio)
        train_size = len(train_full) - val_size
        train_ds, val_ds = random_split(train_full, [train_size, val_size])
    
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)


    def get_train_loader(self):
        return self.train_loader

    def get_val_loader(self):
        return self.val_loader