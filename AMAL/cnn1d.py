import gzip
from tp5_preprocess import TextDataset
import torch
import torch.nn as nn
import torch.optim as optim
import sentencepiece as spm

''' training functions '''
def accuracy(yhat,y):
    predict = torch.argmax(yhat,dim=1)
    return torch.sum(predict==y).float()/len(y)

def train(model,iterator,optimizer,criterion,device,training=False):
    model.eval()
    if training:
        model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for x,y in iterator:
        x,y = x.to(device),y.to(device)
        output = model(x)
        loss = criterion(output,y)
       
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        acc = accuracy(output,y)
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss/len(iterator),epoch_acc/len(iterator)

''' CNN model '''
class CNN1d(nn.Module):
    def __init__(self,vocab_size,emb_size,nb_classes,embeddings,pooling,convo,device="cpu"):
        super(CNN1d,self).__init__()
        self.embeddings = embeddings
        self.device = device
        self.convo_pooling = nn.ModuleList([nn.Conv1d(convo[0][0],convo[1][0],convo[2][0],convo[3][0],convo[4][0]).to(device),
                                            nn.MaxPool1d(pooling[0][0],pooling[1][0],pooling[2][0]).to(device)]) 
        for i in range(1,len(pooling[0])):
            self.convo_pooling.append(nn.Conv1d(convo[0][i],convo[1][i],convo[2][i],convo[3][i],convo[4][i]).to(device))
            if i!=len(pooling[0])-1:
                self.convo_pooling.append(nn.MaxPool1d(pooling[0][i],pooling[1][i],pooling[2][i]).to(device))
                
        self.linear = nn.Linear(convo[1][-1],nb_classes)
    
    def forward(self,x):
        #x : b x l 
        embedded = self.embeddings(x).permute(0,2,1) #b x e x l
        for forward in (self.convo_pooling):
            embedded = forward(embedded) # b x last pooling out x depend on previous
        out,_ = torch.max(embedded,dim=2) # b x last pooling out
        out = self.linear(out)
        return out    

''' data '''
datafile = "train-1000.pth"
with gzip.open(datafile, "rb") as fp:
    train_data = torch.load(fp)
datafile = "test-1000.pth"
with gzip.open(datafile, "rb") as fp:
    test_data = torch.load(fp)
    
''' iterator '''    
batch_size = 256
train_iterator = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, collate_fn=TextDataset.collate, drop_last=True)
test_iterator = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=batch_size, collate_fn=TextDataset.collate, drop_last=True)
  
''' sentencepiece '''
tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(f"wp{vocab_size}.model")
ntokens = len(tokenizer)
  
''' embeddings '''
emb_size = 128
vocab_size = 1000
embeddings = nn.Embedding(vocab_size,emb_size)

''' model hyperparameters '''   
pooling_kernels = [3,3] 
pooling_strides = [2,2]
pooling_paddings = [0,0]
convo_ins = [emb_size,64]
convo_outs = [64,32]
convo_kernels = [3,3]  
convo_strides = [1,1]
convo_paddings = [0,0]
pooling = [pooling_kernels,pooling_strides,pooling_paddings]
convo = [convo_ins,convo_outs,convo_kernels,convo_strides,convo_paddings]

''' model '''
nb_classes = 3  
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN1d(vocab_size,emb_size,nb_classes,embeddings,pooling,convo,device).to(device)

''' hyperparameters '''
nb_epochs = 1
lr = 1e-3
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=lr)

print("Training ...")
for epoch in range(nb_epochs):
    train_loss,train_acc = train(model,train_iterator,optimizer,criterion,device, training=True)
    test_loss,test_acc = train(model,test_iterator,_,criterion,device, training=False)
    print(f'Epoch {epoch+1:02}')
    print(f'\tTrain Loss : {train_loss:.3f}')
    print(f'\tTrain Accuracy : {train_acc:.3f}')    
    print(f'\tTest Accuracy : {test_acc:.3f}')
    
    
    