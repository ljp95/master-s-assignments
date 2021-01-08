
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt

################################################# DATA #################################################

TRANSFORMS = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
        ])

train_data = torchvision.datasets.MNIST("/tmp/mnist", train=True, transform=TRANSFORMS, target_transform=None, download=True)
test_data = torchvision.datasets.MNIST("/tmp/mnist", train=False, transform=TRANSFORMS, target_transform=None, download=True)

BATCH_SIZE = 64
train_iterator = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)  
test_iterator = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)  

################################################# NETWORK #################################################

class VAE(nn.Module):
    def __init__(self,input_size,enc_hidden_size,dec_hidden_size,latent_size,device):
        super(VAE,self).__init__()
        self.latent_size = latent_size
        self.device = device
        self.encoder = nn.Sequential(
                nn.Linear(input_size,enc_hidden_size),
                nn.ReLU(),
                nn.Linear(enc_hidden_size,latent_size*2))
        self.decoder = nn.Sequential(
                nn.Linear(latent_size,dec_hidden_size),
                nn.ReLU(),
                nn.Linear(dec_hidden_size,input_size),
                nn.Sigmoid())
        
    def encode(self,x):
        encoded = self.encoder(x)
        mu, logsigma = encoded[:,:self.latent_size],encoded[:,self.latent_size:]
        eps = torch.randn(x.shape[0],self.latent_size).to(self.device)
        z = mu + torch.exp(logsigma/2) * eps
        return (mu,logsigma,eps),z
    
    def decode(self,z):
        return self.decoder(z)    

    def forward(self,x):
        (mu,logsigma,eps), encoded = self.encode(x)
        decoded = self.decode(encoded)
        return (mu,logsigma,eps),encoded,decoded
    
################################################# PARAMETERS #################################################
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 784
enc_hidden_size = 256
dec_hidden_size = 256
latent_size = 100
model = VAE(input_size,enc_hidden_size,dec_hidden_size,latent_size,device=device).to(device)

lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(),lr = lr)    
bce_fn = torch.nn.BCELoss(reduction='sum')

nb_epochs = 5     

################################################# TRAINING #################################################

for epoch in range(nb_epochs):
    for x,y in train_iterator:
        #input
        x = x.to(device)
        input = x.reshape(x.shape[0],-1)
        
        #forward
        (mu,logsigma,eps),encoded,decoded = model(input)
        
        #loss
        bce_loss = bce_fn(decoded,input)
        kl_loss = -1/2 * torch.mean(torch.sum(1+logsigma-mu**2-torch.exp(logsigma),dim=1))
        loss = bce_loss + kl_loss
        
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #testing
    x,y = test_data[0]
    input = x.to(device).reshape(x.shape[0],-1)
    (mu,logsima,eps),encoded,decoded = model(input)
    plt.figure()
    plt.imshow(decoded.cpu().detach().reshape(28,28).numpy(),cmap="gray")
    
plt.figure()
plt.imshow(x[0],cmap="gray")
        
