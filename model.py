import torch

class mlp(torch.nn.Module):
    
    def __init__(self,inputDim=400,hiddenDim=1024):
        super().__init__()

        self.inputDim = inputDim
        #self.timeSteps = timeSteps
        self.layer_1 = torch.nn.Linear(inputDim*inputDim,hiddenDim)
        self.layer_2 = torch.nn.Linear(hiddenDim,int(hiddenDim/2))
        self.layer_3 = torch.nn.Linear(int(hiddenDim/2),2) #
        #self.layer_3 = torch.nn.Linear(hiddenDim,2*inputDim*timeSteps+timeSteps) #
        self.softPlus = torch.nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):

        x = x.view(self.inputDim*self.inputDim)
        x =self.sigmoid(self.layer_1(x))
        x =self.sigmoid( self.layer_2(x))
        modelParameters = self.layer_3(x)

        mus = modelParameters[0]
        sigs = self.softPlus(modelParameters[1])
        #mus = modelParameters[:self.inputDim*self.timeSteps]
        #sigs = self.softPlus(modelParameters[self.inputDim*self.timeSteps:2*self.inputDim*self.timeSteps])
        #betas = self.sigmoid(modelParameters[2*self.inputDim*self.timeSteps:])

        return mus,sigs#,betas
        
