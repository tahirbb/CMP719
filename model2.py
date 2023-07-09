import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import random
import torch.nn.functional as F
import os
import csv

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.empty_cache()


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1, groups=32),
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64),
            nn.Conv2d(64, 128, kernel_size=1, stride=1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

# Intermediate autoencoder
        self.feedforward = FeedForward(128, 2, bias=True)
        self.feedforward1 = FeedForward(64, 2, bias=True)
#        self.flatten = nn.Flatten()
#        self.latent = nn.Linear(175232, 128)

        # Decoder
        self.decoder1 = nn.Sequential(
#            nn.Linear(128, 175232),
#            nn.Unflatten(1, (128, 37, 37)),
            nn.ConvTranspose2d(128,64, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
#            nn.Linear(128, 175232),
#            nn.Unflatten(1, (128, 37, 37)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )



    def forward(self, x):
        orig = x
        x = self.encoder1(x)
        x = self.feedforward(x)
#        x = self.flatten(x)
#        x = self.latent(x)

        x = self.decoder1(x)
        x = self.encoder(x)
        x = self.feedforward1(x)
        x = self.decoder(x)
        x = x + orig
        x = torch.clamp(x, 0, 1)
        return x

# Additional autoencoder module
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        latent_dim = 1
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


train_loss_values = []  # List to store training loss values
val_loss_values = []  # List to store validation loss values

# Training function
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))
    train_loss_values.append(train_loss)

# Validation function
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)
            output = model(data)
            val_loss += criterion(output, data).item()
    val_loss /= len(val_loader)
    print('====> Validation set loss: {:.4f}'.format(val_loss))
    val_loss_values.append(val_loss)

# Testing function
def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for data, _, filenames in test_loader:
            data = data.to(device)
            output = model(data)
            test_loss += criterion(output, data).item()

            # Save the output images
            for i in range(data.size(0)):
                output_img = output[i].detach().cpu()
                output_img = transforms.ToPILImage()(output_img)
                output_filename = os.path.splitext(filenames[i])[0] + "_output.png"
                output_img.save(os.path.join(output_folder, output_filename))

    test_loss /= len(test_loader)
    print('====> Test set loss: {:.4f}'.format(test_loss))


# Set random seed for reproducibility
torch.manual_seed(2023)

# Dataset paths
blurred_folder = '/home/thr/CMP719/AutoEncoderPart/Blurred'
gt_folder = '/home/thr/CMP719/AutoEncoderPart/Gt'
output_folder = '/home/thr/CMP719/AutoEncoderPart/Out'

# Split dataset into train, validation, and test sets
blurred_filenames = sorted([f for f in os.listdir(blurred_folder) if f.startswith('000')])
gt_filenames = sorted([f for f in os.listdir(gt_folder) if f.startswith('000')])

total_samples = len(blurred_filenames)
train_samples = int(0.7 * total_samples)
val_samples = int(0.1 * total_samples)
test_samples = total_samples - train_samples - val_samples

random.seed(2023)
random.shuffle(blurred_filenames)
random.shuffle(gt_filenames)

train_blurred_filenames = blurred_filenames[:train_samples]
train_gt_filenames = gt_filenames[:train_samples]

val_blurred_filenames = blurred_filenames[train_samples:train_samples + val_samples]
val_gt_filenames = gt_filenames[train_samples:train_samples + val_samples]

test_blurred_filenames = blurred_filenames[train_samples + val_samples:]
test_gt_filenames = gt_filenames[train_samples + val_samples:]

# Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Datasets and DataLoaders
train_dataset = []
val_dataset = []
test_dataset = []

for i in range(train_samples):
    blurred_img = Image.open(os.path.join(blurred_folder, train_blurred_filenames[i]))
    gt_img = Image.open(os.path.join(gt_folder, train_gt_filenames[i]))

    train_dataset.append((transform(blurred_img), transform(gt_img)))

for i in range(val_samples):
    blurred_img = Image.open(os.path.join(blurred_folder, val_blurred_filenames[i]))
    gt_img = Image.open(os.path.join(gt_folder, val_gt_filenames[i]))

    val_dataset.append((transform(blurred_img), transform(gt_img)))


for i in range(test_samples):
    blurred_img = Image.open(os.path.join(blurred_folder, test_blurred_filenames[i]))
    gt_img = Image.open(os.path.join(gt_folder, test_gt_filenames[i]))

    test_dataset.append((transform(blurred_img), transform(gt_img), test_blurred_filenames[i]))


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize the model
model = Autoencoder().to(device)
print(model)


# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, weight_decay = 0.001)


# Training and testing loop
for epoch in range(1, 71):
    train(model, train_loader, criterion, optimizer, epoch)
    validate(model, val_loader, criterion)

    if epoch % 5 == 0:
        filename = f"model_epoch_3Tem_{epoch}.pth"
        torch.save(model.state_dict(), filename)
        print(f"Saved model at epoch {epoch} as {filename}")
                # Save train and validation loss values as CSV files

        train_loss_filename = f"train_loss_epoch_3Tem_{epoch}.csv"
        val_loss_filename = f"val_loss_epoch_3Tem_{epoch}.csv"

        with open(train_loss_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Loss'])
            for i, loss in enumerate(train_loss_values):
                writer.writerow([i + 1, loss])

        with open(val_loss_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Loss'])
            for i, loss in enumerate(val_loss_values):
                writer.writerow([i + 1, loss])

        print(f"Saved train loss values at epoch {epoch} as {train_loss_filename}")
        print(f"Saved validation loss values at epoch {epoch} as {val_loss_filename}")



# Test the model
test(model, test_loader, criterion)

torch.save(model.state_dict(), 'mymodel.pth')

train_loss_filename = 'train_loss_3Tem_values.csv'
val_loss_filename = 'validation_loss_3Tem_values.csv'

with open(train_loss_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Loss'])
    for i, loss in enumerate(train_loss_values):
        writer.writerow([i + 1, loss])


with open(val_loss_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Loss'])
    for i, loss in enumerate(val_loss_values):
        writer.writerow([i + 1, loss])

