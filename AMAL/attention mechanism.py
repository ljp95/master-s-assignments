
import re
from pathlib import Path
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from datamaestro import prepare_dataset
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

class FolderText(Dataset):
    def __init__(self, classes, folder: Path, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = {}
        for ix, key in enumerate(classes):
            self.labels[key] = ix
        
        for label in classes:
            for file in (folder / label).glob("*.txt"):
                self.files.append(file.read_text(encoding="utf-8") if load else file)
                self.filelabels.append(self.labels[label])

    def __len__(self):
        return len(self.filelabels)
    
    def __getitem__(self, ix):
        s = self.files[ix]
        return self.tokenizer(s if isinstance(s, str) else s.read_text(encoding="utf-8")), self.filelabels[ix]
    
def get_imdb_data(embedding_size=50):
    WORDS = re.compile(r"\S+")

    word2id, embeddings = prepare_dataset('edu.stanford.glove.6b.%d' % embedding_size).load()
    OOVID = len(word2id)
    word2id["__OOV__"] = OOVID
    embeddings = np.vstack((embeddings, np.zeros(embedding_size)))

    def tokenizer(t):
        return [word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())]

    logging.info("Loading embeddings")

    logging.info("Get the IMDB dataset")
    ds = prepare_dataset("edu.stanford.aclimdb")

    return word2id, embeddings, FolderText(ds.train.classes, ds.train.path, tokenizer, load=True), FolderText(ds.test.classes, ds.test.path, tokenizer, load=True)

def collate(batch):
    ''' pad the sequences of the batch '''
    X,Y,X_lengths = [],[],[]
    for item in batch:
        Y.append(item[1])
        X.append(torch.LongTensor(item[0]))
        X_lengths.append(len(item[0]))
    X = torch.nn.utils.rnn.pad_sequence(X,padding_value = word2id["__OOV__"]) #l x b
    Y = torch.LongTensor(Y) #b
    X_lengths = torch.tensor(X_lengths)
    mask = torch.arange(X.size(0))[None,:]<X_lengths.view(-1)[:,None]        
    return X,Y,X_lengths,mask 

class naive_model(nn.Module):
    def __init__(self,emb_size,nb_class,embeddings,device="cpu"):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embeddings),freeze=True)
        self.linear = nn.Linear(emb_size,nb_class)
        self.device = device
        
    def forward(self,x,x_lengths,mask=None):
        # x : l x b        
        embedded = self.embedding(x) #l x b x e
        yhat = self.linear(torch.sum(embedded,dim=0)/x_lengths.unsqueeze(1).to(device))
        return yhat,_
    
class attention_model(nn.Module):
    def __init__(self,emb_size,nb_class,embeddings,device="cpu"):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embeddings),freeze=True)
        self.q = nn.Linear(emb_size,1)
        self.linear = nn.Linear(emb_size,nb_class)
        self.device = device        
        
    def forward(self,x,x_lengths,mask):
        # x : l x b
        embedded = self.embedding(x).permute(1,0,2) #b x l x e
        input_softmax = self.q(embedded).squeeze(2) #b x l
        input_softmax = torch.where(mask,input_softmax,(torch.ones_like(mask)*-float("inf")).to(self.device))
        attention = torch.softmax(input_softmax,dim=1).unsqueeze(2) #b x l x 1 
        representation = embedded * attention # b x l x e
        yhat = self.linear(torch.sum(representation,dim=1))
        return yhat,attention.squeeze(2)
      
class attention_modelv2(nn.Module):
    def __init__(self,emb_size,nb_class,embeddings,device="cpu"):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embeddings),freeze=True)
        self.q_linear = nn.Linear(emb_size,emb_size)
        self.emb_linear = nn.Linear(emb_size,emb_size)        
        self.linear = nn.Linear(emb_size,nb_class)
        self.device = device
        
    def forward(self,x,x_lengths,mask):
        # x : l x b
        embedded = self.embedding(x).permute(1,0,2) #b x l x e
        mean_embeddings = torch.mean(embedded,dim=1) #b x e        
        q = self.q_linear(mean_embeddings).unsqueeze(1) # b x 1 x e
        input_softmax = torch.sum(q*embedded,dim=2)
        input_softmax = torch.where(mask,input_softmax,(torch.ones_like(mask)*-float("inf")).to(self.device))
        attention = torch.softmax(input_softmax,dim=1).unsqueeze(2) #b x l x 1 
        embedded_linear = self.emb_linear(embedded)
        representation = embedded_linear * attention # b x l x e
        yhat = self.linear(torch.sum(representation,dim=1))
        return yhat,attention.squeeze(2)
    
def accuracy(yhat,y):
    predict = torch.argmax(yhat,dim=1)
    return torch.sum(predict==y).float()/len(y)

def train(model,iterator,optimizer,criterion,device,training=False):
    model.eval()
    if training:
        model.train()
    epoch_loss = 0
    epoch_acc = 0
    epoch_entropy = []
    
    for x,y,x_lengths,mask in iterator:
        x,y,mask = x.to(device),y.to(device),mask.to(device)
        output,attention = model(x,x_lengths,mask)
        loss = criterion(output,y)
       
        entropy = torch.distributions.categorical.Categorical(probs = attention).entropy()     
        loss += torch.mean(entropy,dim=0)/10 #entropy regularization       
        epoch_entropy.append(entropy.cpu().detach())

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        acc = accuracy(output,y)
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss/len(iterator),epoch_acc/len(iterator),torch.cat(epoch_entropy)

emb_size = 50

''' dataloader '''
batch_size = 32
word2id, embeddings, train_data, test_data = get_imdb_data(emb_size)
train_iterator = DataLoader(train_data,shuffle=True, batch_size=batch_size, collate_fn=collate, drop_last=True)  
test_iterator = DataLoader(test_data,shuffle=False, batch_size=batch_size, collate_fn=collate, drop_last=True) 
nb_class = 2

id2word = {v:k for k,v in word2id.items()}

''' model '''    
device = "cuda" if torch.cuda.is_available() else "cpu"
#model = naive_model(emb_size,nb_class,embeddings,device).to(device)
#model = attention_model(emb_size,nb_class,embeddings,device).to(device)
model = attention_modelv2(emb_size,nb_class,embeddings,device).to(device)

''' hyperparameters '''
nb_epochs = 20
lr = 1e-3
optimizer = optim.Adam(model.parameters(),lr=lr)
criterion = nn.CrossEntropyLoss()

print("Training ...")
for epoch in range(nb_epochs):
    train_loss,train_acc,train_entropy = train(model,train_iterator,optimizer,criterion,device, training=True)
    test_loss,test_acc,test_entropy = train(model,test_iterator,_,criterion,device, training=False)
    plt.hist(train_entropy.numpy())
    print(f'Epoch {epoch+1:02}')
    print(f'\tTrain Loss : {train_loss:.3f}')
    print(f'\tTrain Accuracy : {train_acc:.3f}')    
    print(f'\tTest Accuracy : {test_acc:.3f}')

''' most important words '''

#getting attention of a sentence
cpt = 0
for x,y,x_lengths,mask in test_iterator:
    x,y,mask = x.to(device),y.to(device),mask.to(device)
    output,attention = model(x,x_lengths,mask)
    cpt += 1
    if cpt:
        break
index = 1
one_attention = attention[index][:x_lengths[index]] 

#printing sentence   
print(sentence)

#plotting attention   
plt.plot(one_attention.cpu().detach().numpy())

#getting most important words
nb_words = 10
sorted_attention = torch.argsort(one_attention,descending=True)
for i in range(nb_words):
    nb = x[sorted_attention[i].item(),index].item()
    print(f'{id2word[nb]} : {one_attention[sorted_attention[i].item()]:.3f}')
sentence = ' '.join([id2word[x[i,index].item()] for i in range(x_lengths[index])]) 


