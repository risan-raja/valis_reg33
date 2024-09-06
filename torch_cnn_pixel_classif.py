import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models.resnet import ResNet101_Weights, ResNet18_Weights

class PixelWiseClassifierWithBackbone(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(PixelWiseClassifierWithBackbone, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # Adjust in_channels based on backbone
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv_layers.cuda()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.upsample.cuda()
        # Final 1x1 convolution for classification
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1, device='cuda')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = self.backbone(x)
        x = self.conv_layers(x)
        # print(x.shape)
        x = self.upsample(x)
        # print(x.shape)
        x = self.classifier(x)
        x = self.sigmoid(x)
        return x

# Example usage
model = PixelWiseClassifierWithBackbone(num_classes=1)  # Adjust num_classes as needed
# Define transformations (adjust as needed)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create Training Dataset

class SlideDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        return image, label

train_dataset = SlideDataset(train_images, label_masks, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
wandb.finish()
import wandb
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="PixelClassification",

    # track hyperparameters and run metadata
    config={
    "learning_rate":0.001,
    "architecture": "CNN",
    "dataset": "train_dataset",
    "epochs": 10,
    }
)
from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss, FocalLoss

from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss, FocalLoss
def train_model(model, train_loader, num_epochs=10,):
    model.train()
    criterion = nn.BCELoss()
    dice = DiceLoss(mode='binary')
    jaccard = JaccardLoss(mode='binary')
    focal = FocalLoss(mode='binary')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.squeeze(1)
            outputs = transforms.Resize(images[0].shape[1:])(outputs)
            loss = criterion(outputs.unsqueeze(0), labels)
            dice_loss = dice(outputs.unsqueeze(0), labels)
            jaccard_loss = jaccard(outputs.unsqueeze(0), labels)
            focal_loss = focal(outputs.unsqueeze(0), labels)
            loss = criterion(outputs.unsqueeze(0), labels)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")
                
        wandb.log({
            "dice_loss": dice_loss.item(),
            "jaccard_loss": jaccard_loss.item(),
            "focal_loss": focal_loss.item(),
            "bce_loss": loss.item()
            })


train_model(model, train_loader, num_epochs=10)
# Load an image
image = cv2.imread('train_block_out/msrcr/0507_msrcr.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = transform(image)
image_input = image.unsqueeze(0)  # Add batch dimension
# Forward pass
image_input = image_input.cuda()
with torch.no_grad():  # No need to track gradients during inference
    output = model(image_input)
    threshold = 0.5
    output = (output > threshold).float()
# output = output.cpu().numpy()