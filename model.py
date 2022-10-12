import torch

class mlp(torch.nn.Module):
    
    def __init__(self,timeSteps,inputDim=256,hiddenDim=256):
        super().__init__()

        self.inputDim = inputDim
        self.timeSteps = timeSteps
        self.timeEmbedding = torch.nn.Embedding(timeSteps,2*inputDim)
        self.layer_1 = torch.nn.Linear(2*inputDim,hiddenDim)
        #self.layer_2 = torch.nn.Linear(hiddenDim,hiddenDim)
        self.layer_3 = torch.nn.Linear(hiddenDim,2*inputDim) #
        #self.layer_3 = torch.nn.Linear(hiddenDim,2*inputDim*timeSteps+timeSteps) #
        self.softPlus = torch.nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self,x,t,device):

        x = x.contiguous().view(-1,2*self.inputDim) 
        #tEmb = self.timeEmbedding(torch.tensor(t).to(device))
        tEmb = self.timeEmbedding(t)
        x+=tEmb
        x =self.sigmoid(self.layer_1(x))
        #x =self.sigmoid( self.layer_2(x))
        modelParameters = self.layer_3(x)

        #mus = self.sigmoid(modelParameters[:self.inputDim*self.inputDim].view(self.inputDim,self.inputDim,1))
        mus = modelParameters.view(-1,self.inputDim,2)
        #sigs = self.softPlus(modelParameters[self.inputDim*self.inputDim:]).view(self.inputDim,self.inputDim,1)
        #return mus,sigs#,betas
        return mus
        
class mlpImage(torch.nn.Module):
    
    def __init__(self,timeSteps,inputDim=100,hiddenDim=1024):
        super().__init__()

        self.inputDim = inputDim
        self.timeSteps = timeSteps
        self.timeEmbedding = torch.nn.Embedding(timeSteps,inputDim*inputDim)
        self.layer_1 = torch.nn.Linear(inputDim*inputDim,hiddenDim)
        self.layer_2 = torch.nn.Linear(hiddenDim,hiddenDim)
        self.layer_3 = torch.nn.Linear(hiddenDim,1*inputDim*inputDim) #
        #self.layer_3 = torch.nn.Linear(hiddenDim,2*inputDim*timeSteps+timeSteps) #
        self.softPlus = torch.nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self,x,t,device):

        x = x.view(self.inputDim*self.inputDim) 
        tEmb = self.timeEmbedding(torch.tensor(t).to(device))
        x+=tEmb
        x =self.sigmoid(self.layer_1(x))
        x =self.sigmoid( self.layer_2(x))
        modelParameters = self.layer_3(x)

        #mus = self.sigmoid(modelParameters[:self.inputDim*self.inputDim].view(self.inputDim,self.inputDim,1))
        mus = modelParameters[:self.inputDim*self.inputDim].view(self.inputDim,self.inputDim,1)
        #sigs = self.softPlus(modelParameters[self.inputDim*self.inputDim:]).view(self.inputDim,self.inputDim,1)
        #sigs = self.sigmoid(modelParameters[self.inputDim*self.inputDim:]).view(self.inputDim,self.inputDim,1)

        #mus = modelParameters[:self.inputDim*self.timeSteps]
        #sigs = self.softPlus(modelParameters[self.inputDim*self.timeSteps:2*self.inputDim*self.timeSteps])
        #betas = self.sigmoid(modelParameters[2*self.inputDim*self.timeSteps:])

        #return mus,sigs#,betas
        return mus
        
