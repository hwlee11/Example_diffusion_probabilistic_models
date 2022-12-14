import torch
import argparse
from MakeSwissRoll import swissroll
from model import mlp
import random
import matplotlib.pyplot as plt
import time
import numpy

randseed=42
random.seed(randseed)
numpy.random.seed(randseed)
torch.manual_seed(randseed)
torch.cuda.manual_seed(randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def forwardSampling(x_0,betas,device):

    timeSteps = len(betas)
    mu = torch.sqrt(betas)

    t = int(random.uniform(1,timeSteps))

    alphaTensor = 1-betas
    alphaBars = alphaTensor.cumprod(0)
    prevAlphaBar = alphaBars[t-1]
    alphaBar = alphaBars[t]
    # x^(1)...x^(T) sampling
    eps = torch.normal(mean=torch.zeros(64,2),std=torch.ones(64,2)).to(device)
    xNoise = torch.sqrt(alphaBar)*x_0 + torch.sqrt(1-alphaBar)*eps

    #plt.plot(x_0[:,1],x_0[:,0],'ro')
    #plt.plot(xNoise[:,1],xNoise[:,0],'rs')
    #plt.imshow(x_0,cmap=plt.get_cmap('gray'))
    #plt.imshow(xNoise,cmap=plt.get_cmap('gray'))
    #plt.savefig('noised_data.png',dpi=300)
    #exit()

    sigmaPosterior = betas[t] * (1-prevAlphaBar)/(1-alphaBar)
    muPosterior = 1/(torch.sqrt(alphaTensor[t]))*(xNoise-((1-alphaTensor[t])/(torch.sqrt(1-alphaBar)))*eps)

    #x_0T = torch.zeros(timeSteps-1)
    #for t in range(timeSteps):
    #    x_0T[t] = x_0*torch.prod(betas[:t])
    
    return xNoise,t,muPosterior,sigmaPosterior,eps

def torchForwardSampling(x_0,betaT,betaTensor,t,batchSize,dataSize,device):

    #t = int(random.uniform(1,timeSteps))

    alphaTensor = 1-betaT
    alphaBars = alphaTensor.cumprod(0)
    prevAlphaBar = alphaBars[t-1]
    alphaBar = alphaBars[t]
    # x^(1)...x^(T) sampling
    eps = torch.normal(mean=torch.zeros(batchSize,dataSize,2),std=torch.ones(batchSize,dataSize,2)).to(device)
    xNoise = torch.sqrt(alphaBar).unsqueeze(-1).expand(-1,dataSize).unsqueeze(-1).expand(-1,dataSize,2)*x_0
    xNoise += torch.sqrt(1-alphaBar).unsqueeze(-1).expand(-1,dataSize).unsqueeze(-1).expand(-1,dataSize,2)*eps

    sigmaPosterior = betaT[t] * (1-prevAlphaBar)/(1-alphaBar)
    muPosterior= xNoise-((1-alphaTensor[t].unsqueeze(-1).expand(-1,dataSize).unsqueeze(-1).expand(-1,dataSize,2)))/(torch.sqrt(1-alphaBar.unsqueeze(-1).expand(-1,dataSize).unsqueeze(-1).expand(-1,dataSize,2)))*eps
    muPosterior *= 1/(torch.sqrt(alphaTensor[t].unsqueeze(-1).expand(-1,dataSize).unsqueeze(-1).expand(-1,dataSize,2)))

    return xNoise,muPosterior,sigmaPosterior,eps

def reversedSampling(x,eps,betas,t,dataSize,device):

    """
    z = torch.zeros(100,100,1).to(device)
    if t > 1:
        z = torch.normal(mean=torch.zeros(100,100,1),std=torch.ones(100,100,1)).to(device)
    """
    z = torch.zeros(dataSize,2).to(device)
    if t > 1:
        z = torch.normal(mean=torch.zeros(dataSize,2),std=torch.ones(dataSize,2)).to(device)
    
    alphaTensor = 1-betas
    alphaBars = alphaTensor.cumprod(0)
    alphaBar = alphaBars[t]

    #DDPM style
    sigma = betas[t]*z
    x = (1/torch.sqrt(alphaTensor[t]))*(x-(((1-alphaTensor[t])/torch.sqrt(1-alphaBar))*eps)) + sigma

    return x

def LBLL(mu,sigma,muPosterior,sigmaPosterior,t,beta,timeSteps):
    # Lower bound on log likelihood
    # E_{q}[D_{KL}(q(x^(t-1)|)||p(x^((t-1)|p^(t)))] + H_{q}+H_{q}-H_{p}
    # H term is constant
    KL = torch.log(sigma) - torch.log(sigmaPosterior) + (sigmaPosterior**2 + (muPosterior-mu)**2)/(2*sigma**2+torch.finfo(torch.float32).eps) - 0.5

def DDPMLBLL(predictEps,eps,t,beta,timeSteps,dataSize,batchSize):

    # DDPM style Lower bound on log likelihood
    # E_{q}[D_{KL}(q(x^(t-1)|)||p(x^((t-1)|p^(t)))] + H_{q}+H_{q}-H_{p}
    # H term is constant
    # beta is sigma in DDPM 
    # Loss = || eps_{t} - eps_{theta} ||^2
    #L = torch.norm(eps-predicEps,2)
    #print(predicEps.size(),eps.size())
    #L = torch.nn.functional.mse_loss(predicEps,eps,reduction='sum')
    #L/=100*64*2
    #print(L)

    L = torch.sqrt(((eps-predictEps)**2).sum(2)).sum(-1)
    LVar,L = torch.var_mean(L)

    #print(L,L.size())
    #print(((predictEps-eps)**2).sum()/100)
    #L = torch.norm((eps-predictEps))
    #exit()

    return L,LVar


def train(args):

    #init params
    timeSteps = args.time_steps
    inputDim = args.input_dim
    hiddenDim = args.hidden_dim
    dataSize = args.data_size
    device = torch.device(args.device)
    epochs = 4000
    batchs = 1000
    batchSize =128

    #make train data
    data,t = swissroll(dataSize)
    plt.plot(data[:,0],data[:,1],'ro')
    plt.savefig("256sample.png",dpi=300)
    exit()
    plt.clf()
    data = torch.tensor(data).to(torch.float64)
    """
    swissRoll = torch.zeros(100,100,1)
    for x,y in data:
        x = int(100*(x+1.5)/3)
        y = int(100*(y+1.5)/3)
        swissRoll[x][y][0] = 1.0
    data = swissRoll
    """

    model = mlp(timeSteps,inputDim=dataSize,hiddenDim=hiddenDim)
    lr = 0.01
    optim = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9)
    betas = torch.zeros(timeSteps)
    deltaBetas = 0.0001
    beta = deltaBetas
    for i in range(timeSteps):
        betas[i] = beta
        beta+=deltaBetas
    data = data.to(device)
    betaT = betas.to(device)
    model.to(device).float()

    for epoch in range(epochs):
        epochLoss = 0
        epochLossVar = 0
        model.train()
        for i in range(batchs):
            with torch.no_grad():
                t=torch.randint(1,timeSteps,(batchSize,)).to(device)
                betaList = [betaT[t[idx].item()] for idx in range(batchSize)]
                betaTensor = torch.tensor(betaList).float()
                xNoise,muPosterior,sigmaPosterior,eps = torchForwardSampling(data.float(),betaT,betaTensor,t,batchSize,dataSize,device)
            optim.zero_grad()
            #mu = model(xNoise.float(),t,device)
            predicEps = model(xNoise,t,device)
            #loss = LBLL(mu,sigma,muPosterior,sigmaPosterior,timeSteps)
            loss,lossVar = DDPMLBLL(predicEps,eps,t,betaT[t],timeSteps,dataSize,batchSize)
            epochLoss+=loss.item()
            epochLossVar += lossVar.item()
            loss.backward()
            optim.step()

        print('epoch: ',epoch,'loss : ',epochLoss/batchs,'loss var : ',epochLossVar/batchs)
        fileName = "epoch%d_loss%0.1f.dict"%(epoch,epochLoss/batchs)
        torch.save(model.state_dict(),fileName)
        model.eval()
        with torch.no_grad():
            x_t = torch.normal(mean=torch.zeros(dataSize,2),std=torch.ones(dataSize,2)).to(device)
            for t in reversed(range(timeSteps)):
                eps = model(x_t,torch.tensor([t]).to(device),device)
                x_t = reversedSampling(x_t,eps,betaT,t,dataSize,device)
        x = x_t.squeeze(0).to('cpu').numpy()
        fileName = "epoch%d_loss%0.2f_samplingResult.png"%(epoch,epochLoss/batchs)
        #plt.imshow(x,cmap=plt.get_cmap('gray'))
        plt.plot(x[:,0],x[:,1],'rs')
        plt.savefig(fileName,dpi=300)
        plt.clf()

        if epoch%1000 == 0:
            optim.param_groups[0]['lr'] = lr*0.1


def main(args):
    train(args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--time_steps',default=200,type=int)
    parser.add_argument('--input_dim',default=64,type=int)
    parser.add_argument('--data_size',default=128,type=int)
    parser.add_argument('--hidden_dim',default=96,type=int)
    parser.add_argument('--device',default="cuda:0",type=str)
    args = parser.parse_args()
    main(args)
