import logging
import string
from torch.utils.data import Dataset, DataLoader
import unicodedata
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
logging.basicConfig(level=logging.INFO)

PAD = 0
EOS = 1
LETTRES = string.ascii_letters + string.punctuation + string.digits + ' '
id2lettre = dict(zip(range(2,len(LETTRES)+2),LETTRES))
id2lettre[0] = ''
id2lettre[1] = 'END'
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    return ''.join(c for c in unicodedata.normalize('NFD',s) if c in LETTRES)

def string2code(s):
    return [lettre2id[c] for c in normalize(s)]

def code2string(t):
    if type(t) != list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

class TextDataset(Dataset):
    def __init__(self, text: str, *, maxsent=None, maxlen=None):
        self.sentences = [torch.LongTensor(string2code(sentence.strip())+[EOS]) for sentence in text.split('.')]
        
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        return self.sentences[i], len(self.sentences[i])

def collate_fn(samples: List[List[int]]):    
    sentences,lengths = zip(*samples)
    padded_batch = torch.zeros(max(lengths), len(samples)).long() # l x b
    for i in range(len(samples)):
        padded_batch[:lengths[i], i] = sentences[i]
    return padded_batch
    
class LSTM(nn.Module):
    def __init__(self,dico_size,emb_size,latent_size,device='cpu'):
        super(LSTM,self).__init__()
        self.embedding = nn.Embedding(dico_size,emb_size)        
        self.gate_f = nn.Linear(emb_size+latent_size, latent_size)
        self.gate_i = nn.Linear(emb_size+latent_size, latent_size)
        self.gate_o = nn.Linear(emb_size+latent_size, latent_size)
        self.gate_c = nn.Linear(emb_size+latent_size, latent_size)
        self.act = nn.Sigmoid()
        self.decoder = nn.Linear(latent_size, dico_size)
        self.device = device
        
    def one_step(self,embedded,h):
        # embedded : b x e
        # h : b x latent size
        # out : b x latent size
        entry = torch.cat((embedded,h),dim=1)
        f = self.act(self.gate_f(entry))
        i = self.act(self.gate_i(entry))
        self.c = f*self.c + i*torch.tanh(self.gate_c(entry))
        o = self.act(self.gate_o(entry))
        out = o*torch.tanh(self.c)
        return out
        
    def forward(self,x,h):
        # x : l x b
        # h : b x latent_size
        # out : l x b x latent_size
        l = []
        embedded = self.embedding(x)
        self.c = torch.zeros(h.shape).to(self.device)        
#        self.Ct = torch.zeros((x.shape[0],self.D_hidden)).to(device)
        for i in range(x.shape[0]):
            h = self.one_step(embedded[i],h)
            l.append(h)
        return torch.stack(l)
    
    def decode(self, h):
        # h : b x latent size
        return self.decoder(h) # b x dico size
      
class GRU(nn.Module):
    def __init__(self,dico_size,emb_size,latent_size):
        super(GRU,self).__init__()
        self.embedding = nn.Embedding(dico_size,emb_size)
        self.latent_size = latent_size
        self.gate_z = nn.Linear(emb_size+latent_size, latent_size)
        self.gate_r = nn.Linear(emb_size+latent_size, latent_size)
        self.gate_h = nn.Linear(emb_size+latent_size, latent_size)
        self.linear = nn.Linear(latent_size,dico_size)
        self.act = nn.Sigmoid()
        self.decoder = nn.Linear(latent_size, dico_size)

    def one_step(self,embedded,h):
        # embedded : b x e
        # h : b x latent size
        # out : b x latent size        
        entry = torch.cat((embedded,h),dim=1)
        z = self.act_gate(self.gate_z(entry))
        r = self.act_gate(self.gate_r(entry))
        return (1-z)*h + z*torch.tanh(self.gate_h(torch.cat((embedded,h*r),dim=1)))

    def forward(self,x,h):
        # x : l x b
        # h : b x latent_size
        # out : l x b x latent_size        
        l = []
        embedded = self.embedding(x)
        for i in range(x.shape[0]):
            h = self.one_step(embedded[i],h)
            l.append(h)
        return torch.stack(l)
                
    def decode(self, h):
        # h : b x latent size
        return self.decoder(h) # b x dico size
    
def train(model,iterator,optimizer,criterion,device,clip=1):
    model.train()
    epoch_loss = 0

    for x in iterator:
        batch_loss = 0
        h = torch.zeros(batch_size, latent_size).to(device)
        x = x.to(device)
        latents = model.forward(x[:-1],h)
        for i in range(x.shape[0]-1):
            yhat = model.decode(latents[i])
            loss = criterion(yhat,x[i+1])
            batch_loss += loss
            epoch_loss += loss.item()
        batch_loss /= (x.shape[0]-1)
#        print(batch_loss)
        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
            
    return epoch_loss/len(iterator)    

def generate(model,random_gen=True,eos=EOS,start="",maxlen=200,device='cpu'):
    with torch.no_grad():
        model.to(device)
        if start=="":
            sentence = id2lettre[np.random.choice(range(28,54))]
    
        x = torch.LongTensor(string2code(sentence)).view(-1,1).to(device)
        h = torch.zeros(1,latent_size).to(device)
        h = model.forward(x, h)[-1] # 1 x latent_size
        softmax = nn.Softmax(dim=1)
    
        for i in range(maxlen):
            yhat = model.decode(h) 
            if random_gen:
                probas = softmax(yhat).cpu().view(-1).detach().numpy()
                ind = torch.tensor(np.random.choice(len(probas), p=probas)).view(1).to(device)
            else:
                ind = yhat.argmax().view(1).to(device) 
            
            if ind==eos:
                break
            
            sentence += code2string(ind)    
            h = model.one_step(model.embedding(ind),h)
    return sentence

def generate_beam(model,emb_size,k=3,start="",argmax=False,max_len=200,device='cpu'):
    with torch.no_grad():

        model.to(device)
        model.eval()
        softmax = nn.Softmax(dim=1)
        
        # init entries
        if start=="":
            start = id2lettre[np.random.choice(range(28,54))]
        x = torch.LongTensor(string2code(start)).unsqueeze(1).to(device) # 1 x 1
        h = torch.zeros(1,latent_size).to(device) # 1 x latent_size
        h = model.forward(x, h)[-1] # 1 x latent_size
        
        #decoding and tracking
        yhat = model.decode(h) # 1 x dico_size
        
        indices = yhat.argsort(descending=True)[0,:k] # k
        k_probas = torch.log(softmax(yhat)[0,indices]).unsqueeze(1) # k x 1
        
        h = torch.cat(k*[h]) # k x latent_size
        sentence_id = indices.unsqueeze(0) # 1 x k
        
        for i in range(max_len):
            h = model.one_step(model.embedding(sentence_id[-1]),h) # k x latent_size
            yhat = model.decode(h) # k x dico_size
            
            if argmax:
                kxk_indices = yhat.argsort(descending=True)[:,:k].reshape(-1) # k*k
            else:                
                kxk_indices = torch.zeros(k,k)
                probas = softmax(yhat)
                kxk_indices = torch.multinomial(probas,num_samples=k)
                kxk_indices = kxk_indices.reshape(-1) # k x k
                
            arange = torch.arange(k).repeat_interleave(k) # k*k
            kxk_probas = (k_probas + torch.log(softmax(yhat)[arange,kxk_indices]).reshape(k,k)).reshape(k*k) # k x k
        
            indices = kxk_probas.argsort(descending=True)[:k] # k
            k_probas = kxk_probas[indices] # k

            new_h = h.permute(1,0).repeat_interleave(k,dim=1).permute(1,0)
            h = new_h[indices]
            sentence_id = sentence_id.repeat_interleave(k,dim=1)[:,indices]
            sentence_id = torch.cat((sentence_id,kxk_indices[indices].unsqueeze(0)))
        
        sentence = []
        for i in range(k):
            sentence.append((start + code2string(sentence_id[:,i])).split("|")[0])
    return sentence
     
''' dataset '''        
with open ("trump_full_speech.txt","r") as f:
    text = f.read()
dico_size = len(id2lettre)    

''' iterator ''' 
batch_size = 128   
train_iterator = DataLoader(TextDataset(text), collate_fn=collate_fn, batch_size=batch_size, shuffle=True, drop_last=True)

''' hyperparameters '''
emb_size = 64
latent_size = 32
lr = 1e-3

''' model '''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM(dico_size, emb_size, latent_size, device).to(device)

''' objective '''
optimizer = optim.Adam(model.parameters(),lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index=PAD)
nb_epochs = 5

print("Training ...")
for epoch in range(nb_epochs):
    train_loss = train(model,train_iterator,optimizer,criterion,device,clip=1)
    print(f'Epoch {epoch+1:02}')
    print(f'\tTrain Loss : {train_loss:.3f}')
    sentences = generate_beam(model,emb_size,max_len=100,device=device)
    for i in range(len(sentences)):
        print('\t' + sentences[i])




