import torch
import argparse
from MakeSwissRoll import swissroll
from model import mlp
import random
import matplotlib.pyplot as plt

def forwardSampling(x_0,betas):

    timeSteps = len(betas)
    mu = torch.sqrt(betas)

    t = int(random.uniform(1,timeSteps))

    alphaTensor = 1-betas
    alphaBars = alphaTensor.cumprod(0)
    prevAlphaBar = alphaBars[t-1]
    alphaBar = alphaBars[-1]
    # x^(1)...x^(T) sampling
    #xNoise = torch.sqrt(alphaBar)*x_0 + torch.sqrt(1-alphaBar)*torch.normal(mean=torch.zeros(1,400,400),std=torch.ones(1,400,400))
    eps = torch.normal(mean=torch.zeros(400,400,1),std=torch.ones(400,400,1))
    xNoise = torch.sqrt(alphaBar)*x_0 + torch.sqrt(1-alphaBar)*eps

    #plt.plot(x_0[:,1],x_0[:,0],'ro')
    #plt.plot(xNoise[:,1],xNoise[:,0],'rs')
    #plt.imshow(x_0,cmap=plt.get_cmap('gray'))
    #plt.imshow(xNoise,cmap=plt.get_cmap('gray'))
    #plt.savefig('noised_data.png',dpi=300)

    sigmaPosterior = betas[t] * (1-prevAlphaBar)/(1-alphaBar)
    muPosterior = 1/(torch.sqrt(alphaTensor[t]))*(xNoise-((1-alphaTensor[t])/(torch.sqrt(1-alphaBar)))*eps)

    #x_0T = torch.zeros(timeSteps-1)
    #for t in range(timeSteps):
    #    x_0T[t] = x_0*torch.prod(betas[:t])
        
    
    return xNoise,t,muPosterior,sigmaPosterior

def LBLL(mu,sigma,muPosterior,sigmaPosterior):
    return -1


def train(args):

    #init params
    timeSteps = args.time_steps
    inputDim = args.input_dim
    hiddenDim = args.hidden_dim
    epochs = 10
    batchs = 100

    #make train data
    data,t = swissroll()
    swissRoll = torch.zeros(400,400,1)
    for x,y in data:
        x = int(400*(x+2.0)/4)
        y = int(400*(y+2.0)/4)
        swissRoll[x][y][0] = 1.0
    data = swissRoll

    model = mlp(inputDim=inputDim,hiddenDim=hiddenDim)
    betas = torch.zeros(timeSteps)
    deltaBetas = 1/timeSteps
    beta = deltaBetas
    for i in range(timeSteps):
        betas[i] = beta
        beta+=deltaBetas
    xNoise,t,muPosterior,sigmaPosterior = forwardSampling(data,betas)
    mu,sigma = model(xNoise)
    print(mu,sigma)
    #loss = LBLL(mu,sigma,muPosterior,sigmaPosterior)
    exit()

    for epoch in range(epochs):
        for i in range(batchs):
            xNoise,t,muPosterior,sigmaPosterior = forwardSampling(data,betas)
            mu,sigma = model(xNoise)
            loss = LBLL(mu,sigma,muPosterior,sigmaPosterior)

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
    parser.add_argument('--input_dim',default=400,type=int)
    parser.add_argument('--hidden_dim',default=1024,type=int)
    args = parser.parse_args()
    main(args)
