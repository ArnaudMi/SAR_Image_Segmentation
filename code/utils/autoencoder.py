import numpy as np
import os
import torch.nn as nn
import torch
# Définir une classe Dataset personnalisée pour les données complexes
class MonDataset(Dataset):
    def __init__(self, img, dim_height, dim_length):
        self.dim_height = dim_height
        self.dim_length = dim_length
        self.img = img
        self.random_i = [
            int(np.random.random() * (self.height - 400)) for i in range(100)
        ]
        self.random_j = [
            int(np.random.random() * (self.length - 400)) for i in range(100)
        ]
        self.list_img = [
            self.img[
                self.random_i[i] : self.random_i[i] + self.dim_height,
                self.random_j[i] : self.random_j[i] + self.dim_length,
            ]
            for i in range(100)
        ]
        self.height, self.length, self.dim = img.shape

    def __len__(self):
        return len(self.list_img)

    def __getitem__(self, idx):
        i, j = idx // self.length, idx % self.length
        tableau = self.img[i : i + self.dim_height, j : j + self.dim_length]
        tensor = torch.from_numpy(tableau)
        return tensor.type(torch.complex128)  # Convertir le tensor en complex128


# Architecture de l'autoencodeur pour les données complexes
class Autoencodeur(nn.Module):
    def __init__(self):
        super(Autoencodeur, self).__init__()
        self.encodeur = nn.Sequential(
            ComplexConv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ComplexConv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ComplexConv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ComplexConv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.decodeur = nn.Sequential(
            ComplexConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            ComplexConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            ComplexConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            ComplexConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encodeur(x)
        x = self.decodeur(x)
        return x


# Définir une couche de convolution complexe
class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ComplexConv2d, self).__init__()
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        xr, xi = torch.real(x), torch.imag(x)
        yr = self.conv_r(xr) - self.conv_i(xi)
        yi = self.conv_i(xr) + self.conv_r(xi)
        return torch.complex(yr, yi)


# Définir une couche de convolution transposée complexe
class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ComplexConvTranspose2d, self).__init__()
        self.convT_r = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride
        )
        self.convT_i = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride
        )

    def forward(self, x):
        xr, xi = torch.real(x), torch.imag(x)
        yr = self.convT_r(xr) - self.convT_i(xi)
        yi = self.convT_i(xr) + self.convT_r(xi)
        return torch.complex(yr, yi)


# Chemin vers le dossier contenant les fichiers .npy
dossier_dataset = "chemin/vers/votre/dossier"

# Créer une instance du Dataset personnalisé
dataset = MonDataset(dossier_dataset)

# Utiliser un DataLoader pour itérer sur les éléments du Dataset
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Définir votre modèle d'autoencodeur pour les données complexes
model = Autoencodeur()

# Définir la fonction de perte (par exemple, l'erreur quadratique moyenne)
criterion = nn.MSELoss()

# Définir l'optimiseur (par exemple, Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Boucle d'entraînement
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    running_loss = 0.0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

# Une fois l'entraînement terminé, vous pouvez utiliser le modèle pour débruiter des images complexes
# par exemple, en utilisant model(image_bruitee) pour obtenir l'image débruitée
