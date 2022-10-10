import torch
import argparse
from MakeSwissRoll import swissroll
from model import mlp
import random
import matplotlib.pyplot as plt
import time

def forwardSampling(x_0,betas,device):

    timeSteps = len(betas)
    mu = torch.sqrt(betas)

    t = int(random.uniform(1,timeSteps))

    alphaTensor = 1-betas
    alphaBars = alphaTensor.cumprod(0)
    prevAlphaBar = alphaBars[t-1]
    alphaBar = alphaBars[t]
    # x^(1)...x^(T) sampling
    eps = torch.normal(mean=torch.zeros(256,2),std=torch.ones(256,2)).to(device)
    #eps = torch.normal(mean=torch.zeros(100,100,1),std=torch.ones(100,100,1)).to(device)
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

#def reversedSampling(x,mu,betas,t,device):
def reversedSampling(x,eps,betas,t,device):

    """
    z = torch.zeros(100,100,1).to(device)
    if t > 1:
        z = torch.normal(mean=torch.zeros(100,100,1),std=torch.ones(100,100,1)).to(device)
    """
    z = torch.zeros(256,2).to(device)
    if t > 1:
        z = torch.normal(mean=torch.zeros(256,2),std=torch.ones(256,2)).to(device)
    
    alphaTensor = 1-betas
    alphaBars = alphaTensor.cumprod(0)
    alphaBar = alphaBars[t]

    sigma = betas[t]*z
    #eps = torch.normal(mean=mu,std=torch.ones(100,100,1))
    #eps = torch.normal(mean=mu,std=torch.ones(100,100,1))
    #x = (1/torch.sqrt(alphaTensor[t]))*(x-((torch.sqrt(1-alphaTensor[t]))*mu)) + sigma

    x = (1/torch.sqrt(alphaTensor[t]))*(x-(((1-alphaTensor[t])/torch.sqrt(1-alphaTensor[t]))*eps)) + sigma
    #DDPM style
    #x = (1/torch.sqrt(alphaTensor[t]))*(x-((1-alphaTensor[t]/torch.sqrt(1-alphaBar))*torch.normal(mean=mu,std=betas[t])))

    return x

def LBLL(mu,sigma,muPosterior,sigmaPosterior,t,beta,timeSteps):
    # Lower bound on log likelihood
    # E_{q}[D_{KL}(q(x^(t-1)|)||p(x^((t-1)|p^(t)))] + H_{q}+H_{q}-H_{p}
    # H term is constant
    KL = torch.log(sigma) - torch.log(sigmaPosterior) + (sigmaPosterior**2 + (muPosterior-mu)**2)/(2*sigma**2+torch.finfo(torch.float32).eps) - 0.5

#def DDPMLBLL(muPosterior,mu,t,beta,timeSteps):
def DDPMLBLL(predicEps,eps,t,beta,timeSteps):

    # DDPM style Lower bound on log likelihood
    # E_{q}[D_{KL}(q(x^(t-1)|)||p(x^((t-1)|p^(t)))] + H_{q}+H_{q}-H_{p}
    # H term is constant
    # beta is sigma in DDPM 
    #L = (sigmaPosterior**2 + (muPosterior-mu)**2)/(2*sigma**2+torch.finfo(torch.float32).eps) 
    #L = (1/(2*beta**2))*torch.norm(muPosterior-mu,2)
    L = torch.norm(eps-predicEps,2)
    #KL = (KL*timeSteps).mean()
    #print(KL)

    return L


def train(args):

    #init params
    timeSteps = args.time_steps
    inputDim = args.input_dim
    hiddenDim = args.hidden_dim
    device = torch.device(args.device)
    epochs = 100
    batchs = 10000
    batchSize =100

    #make train data
    data,t = swissroll()
    data = torch.tensor(data).to(torch.float64)
    """
    swissRoll = torch.zeros(100,100,1)
    for x,y in data:
        x = int(100*(x+1.5)/3)
        y = int(100*(y+1.5)/3)
        swissRoll[x][y][0] = 1.0
    data = swissRoll
    """

    model = mlp(timeSteps,inputDim=inputDim,hiddenDim=hiddenDim)
    lr = 0.01
    optim = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9)
    betas = torch.zeros(timeSteps)
    deltaBetas = 1/(timeSteps+1)
    beta = deltaBetas
    for i in range(timeSteps):
        betas[i] = beta
        beta+=deltaBetas
    data = data.to(device)
    betas = betas.to(device)
    model.to(device).float()
    #TEST
    #xNoise,t,muPosterior,sigmaPosterior = forwardSampling(data,betas)
    #mu,sigma = model(xNoise)
    #loss = LBLL(mu,sigma,muPosterior,sigmaPosterior)

    for epoch in range(epochs):
        epochLoss = 0
        model.train()
        for i in range(batchs):
            """
            with torch.no_grad():
                xList = list()
                tList = list()
                for j in range(batchSize):
                    xNoise,t,muPosterior,sigmaPosterior,eps = forwardSampling(data,betas,device)
                    xList.append(xNoise)
                    tList.append(t)
            x = torch.stack(xList)
            t = torch.tensor(tList).to(device)
            print(x,x.size())
            print(t)
            exit()
            """
            with torch.no_grad():
                xNoise,t,muPosterior,sigmaPosterior,eps = forwardSampling(data,betas,device)
            #print('data',xNoise)
            #print('smaple',muPosterior,sigmaPosterior)
            optim.zero_grad()
            #mu,sigma = model(xNoise)
            #mu = model(xNoise.float(),t,device)
            predicEps = model(xNoise.float(),t,device)
            #print('mu',mu)
            #print(mu[0],sigma[0])
            #loss = LBLL(mu,sigma,muPosterior,sigmaPosterior,timeSteps)
            loss = DDPMLBLL(predicEps,eps,t,betas[t],timeSteps)
            #loss = DDPMLBLL(muPosterior,mu,t,betas[t],timeSteps)
            epochLoss+=loss.item()
            loss.backward()
            optim.step()

        print('epoch: ',epoch,'loss : ',epochLoss/batchs)
        fileName = "epoch%d_loss%0.1f.dict"%(epoch,epochLoss/batchs)
        torch.save(model.state_dict(),fileName)
        model.eval()
        with torch.no_grad():
            x_t = torch.normal(mean=torch.zeros(256,2),std=torch.ones(256,2)).to(device)
            #x_t = torch.normal(mean=torch.zeros(100,100,1),std=torch.ones(100,100,1)).to(device)
            for t in reversed(range(timeSteps)):
                eps = model(x_t,t,device)
                x_t = reversedSampling(x_t,eps,betas,t,device)
        x = x_t.to('cpu').numpy()
        fileName = "epoch%d_loss%0.1f_samplingResult.png"%(epoch,epochLoss/batchs)
        #plt.imshow(x,cmap=plt.get_cmap('gray'))
        plt.plot(x[:,1],x[:,0],'rs')
        plt.savefig(fileName,dpi=300)

        if epoch%25 == 0:
            optim.param_groups[0]['lr'] = lr*0.1



    """
        x_0 = torch.rand(2)
        #mus,sigs,betas = model(x_0)
        mus,sigs = model(x_0)
        xForward = forwardSampling(x_0,betas)

        #for t in range(timeStesp):
    """


def main(args):
    train(args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--time_steps',default=50,type=int)
    parser.add_argument('--input_dim',default=256,type=int)
    parser.add_argument('--hidden_dim',default=256,type=int)
    parser.add_argument('--device',default="cuda:0",type=str)
    args = parser.parse_args()
    main(args)
