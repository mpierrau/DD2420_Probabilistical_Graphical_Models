## Code for Metropolis Hastings Part of  
# Tutorial 7 in DD2635 or w/e , PGM

## Here we implement the Metropolis-Hastings algorithm from scratch
## IT SHOULD WORK FOR THE GENERAL CASE

from random import uniform
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def gaussMixDensity(x,a1,mu1,sig1,a2,mu2,sig2):
    # Implementation of Gaussian mixture model density function
    if(a1 + a2 != 1):
        #print("Error: a1 and a2 must sum to 1.0.")
        return np.NaN
    else:
        gauss1 = norm.pdf(x,mu1,sig1)
        gauss2 = norm.pdf(x,mu2,sig2)
        return a1*gauss1 + a2*gauss2

def sampleGaussMix(a1,mu1,sig1,a2,mu2,sig2,nSamples=1):
    sampleArray = np.empty(nSamples)

    for i in range(nSamples):
        u = uniform(0,1)
        if(u < a1):
            sampleArray[i] = np.random.normal(mu1,sig1)
        else:
            sampleArray[i] = np.random.normal(mu2,sig2)
    
    return sampleArray

def acceptFunc(xi,xStar,p,q):
    # Acceptance function for MH-algorithm
    # p is the target distribution (must be a proper
    # probability distribution taking one parameter, input x)
    # q(x1,x2) is the conditional proposal distribution. 
    # it finds density of x1 conditioned on x2 as mean. 
    # q does not have to be a normalized distribution, 
    # 
    # xi is the current state
    # xStar is the sampled x to be accepted with probability acceptProb

    acceptProb = np.min([1,(p(xStar)*q(xi,xStar))/(p(xi)*q(xStar,xi))])
    return acceptProb

def metropolisHastings(N,thisTargetDistr, thisPropDens, thisPropSample, thisAcceptFunc):
    # Description of MH-algo

    x = np.empty(N+1) # Preallocate x
    x[0] = 1        # Initialize x0  
    
    for i in range(N):
        u = uniform(0,1)
        xStar = thisPropSample(x[i])
        if u < thisAcceptFunc(x[i],xStar,thisTargetDistr,thisPropDens):
            x[i+1] = xStar  # Accepted
        else:
            x[i+1] = x[i]   # Rejected

    return x


# TODO: figure out how to initialize xStar and sample from q
steps = 10000

a1 = 0.5    # Parameters for target distribution
a2 = 0.5
mu1 = 0
mu2 = 3
sig1 = 1
sig2 = 0.5

mu = 0      # Parameters for proposal distribution q

pDensity = lambda x: gaussMixDensity(x,a1,mu1,sig1,a2,mu2,sig2)
MHaccept = lambda xi, xStar, p, q: acceptFunc(xi,xStar, p, q)
sig = [i for i in np.linspace(1,3,5)]

gaussMixSamples = sampleGaussMix(a1,mu1,sig1,a2,mu2,sig2,10000)
gaussMixSampleProbs = gaussMixDensity(gaussMixSamples,a1,mu1,sig1,a2,mu2,sig2)

def plotMH(sig):    
    for i in range(0,len(sig)):
        qDensity = lambda x, mu: norm.pdf(x,mu,sig[i])
        qSample = lambda mu: np.random.normal(mu,sig[i]) #??

        xSamples = metropolisHastings(steps, pDensity, qDensity, qSample, MHaccept)

        plt.clf()
        plt.hist(xSamples, bins=100, density = True, alpha=0.4)
        plt.plot(gaussMixSamples,gaussMixSampleProbs,'r.',alpha=0.4,markersize=0.4)
        plt.xlabel('x')
        plt.ylabel('p(x)')
        plt.legend(['PDF','Samples'],markerscale=2)
        plt.savefig('tut7_mh_var_' + str(sig[i]) + '.png')
        #plt.show()
