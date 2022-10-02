import torch
import argparse
from MakeSwissRoll import swissroll
from model import mlp

def forwardSampling(x_0,betas):

    print(betas)
    timeSteps = len(betas)
    mu = torch.sqrt(betas)

    # x^(1)...x^(T) sampling
    x_0T = torch.zeros(timeSteps-1)
    for t in range(timeSteps):
        x_0T[t] = x_0*torch.prod(betas[:t])
        


def train(args):

    #init params
    timeSteps = args.time_steps
    inputDim = args.input_dim
    hiddenDim = args.hidden_dim
    epochs = 1

    #make train data
    data,t = swissroll()
    model = mlp(inputDim=inputDim,timeSteps=timeSteps,hiddenDim=hiddenDim)
    betas = torch.zeros(timeSteps)

    for epoch in range(epochs):
        x_0 = torch.rand(2)
        #mus,sigs,betas = model(x_0)
        mus,sigs = model(x_0)
        xForward = forwardSampling(x_0,betas)

        #for t in range(timeStesp):
    


def main(args):
    train(args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--time_steps',default=50,type=int)
    parser.add_argument('--input_dim',default=2,type=int)
    parser.add_argument('--hidden_dim',default=3,type=int)
    args = parser.parse_args()
    main(args)
