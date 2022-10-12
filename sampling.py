import torch
import argparse
from MakeSwissRoll import swissroll
from model import mlp
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

def reversedSampling(x,eps,betas,t,device):

    """
    z = torch.zeros(100,100,1).to(device)
    if t > 1:
        z = torch.normal(mean=torch.zeros(100,100,1),std=torch.ones(100,100,1)).to(device)
    """
    z = torch.zeros(64,2).to(device)
    if t > 1:
        z = torch.normal(mean=torch.zeros(64,2),std=torch.ones(64,2)).to(device)
    
    alphaTensor = 1-betas
    alphaBars = alphaTensor.cumprod(0)
    alphaBar = alphaBars[t]

    sigma = betas[t]*z
    #eps = torch.normal(mean=mu,std=torch.ones(100,100,1))
    #eps = torch.normal(mean=mu,std=torch.ones(100,100,1))
    #x = (1/torch.sqrt(alphaTensor[t]))*(x-((torch.sqrt(1-alphaTensor[t]))*mu)) + sigma

    x = (1/torch.sqrt(alphaTensor[t]))*(x-(((1-alphaTensor[t])/torch.sqrt(1-alphaBar))*eps)) + sigma
    #DDPM style
    #x = (1/torch.sqrt(alphaTensor[t]))*(x-((1-alphaTensor[t]/torch.sqrt(1-alphaBar))*torch.normal(mean=mu,std=betas[t])))

    return x
def sampling(args):

    #init params
    timeSteps = args.time_steps
    inputDim = args.input_dim
    hiddenDim = args.hidden_dim
    modelFilePath = args.model_path
    device = torch.device(args.device)

    model = mlp(timeSteps,inputDim=inputDim,hiddenDim=hiddenDim)
    model.load_state_dict(torch.load(modelFilePath))
    betas = torch.zeros(timeSteps)
    deltaBetas = 0.0001
    beta = deltaBetas
    for i in range(timeSteps):
        betas[i] = beta
        beta+=deltaBetas
    betaT = betas.to(device)
    model.to(device).float()

    model.eval()
    with torch.no_grad():
        x_t = torch.normal(mean=torch.zeros(64,2),std=torch.ones(64,2)).to(device)
        #x_t = torch.normal(mean=torch.zeros(100,100,1),std=torch.ones(100,100,1)).to(device)
        numOfFrame = 0
        for t in reversed(range(timeSteps)):
            eps = model(x_t,torch.tensor(t).to(device),device)
            x_t = reversedSampling(x_t,eps,betaT,t,device)

            x = x_t.squeeze(0).to('cpu').numpy()
            fileName = "model0.1v_frame%d_%d_sampling.png"%(numOfFrame,t)
            plt.title("T = %d"%(t))
            plt.plot(x[:,1],x[:,0],'.')
            plt.savefig(fileName,dpi=300)
            plt.clf()
            numOfFrame+=1


def main(args):
    sampling(args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--time_steps',default=200,type=int)
    parser.add_argument('--input_dim',default=64,type=int)
    parser.add_argument('--hidden_dim',default=64,type=int)
    parser.add_argument('--device',default="cuda:0",type=str)
    parser.add_argument('--model_path',default="./models/epoch2083_loss86.3.dict",type=str)
    args = parser.parse_args()
    main(args)
