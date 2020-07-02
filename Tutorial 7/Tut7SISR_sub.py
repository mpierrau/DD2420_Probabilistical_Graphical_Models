from random import uniform
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx

def SISR(mu_v=0,var_v=10,mu_w=0,var_w=1,priorMu=0,priorVar=10,N=100,T=200):

    ## sequential importance sampling with bootstrap filter 

    def sampleX(xPrev,t,sig):
        tmpMu = 1/2*xPrev + 25*xPrev/(1 + xPrev**2) + 8*math.cos(1.2*t)
        return np.random.normal(tmpMu,sig)

    def computeWeights(xt,sig):
        yt = (xt**2)/20
        return norm.pdf(yt,xt,sig)

    sig_v = np.sqrt(var_v)
    sig_w = np.sqrt(var_w)
    priorSig = np.sqrt(priorVar)

    ## initialization step
    x = np.zeros((N,T))
    x[:,0] = [np.random.normal(priorMu,priorSig) for _ in range(0,N)]
    
    w = np.zeros((N,T))

    w[:,0] = np.repeat(1/N,N)
    tmpW = 1/N
    selected = np.random.choice(range(0,N),size=N,replace=True,p=w[:,0])
    for t in range(1,T):
        
        #Importance Sampling step:        
        for i in range(0,N):
            x[i,t] = sampleX(x[selected[i],t-1],t,sig_v)
            w[i,t] = computeWeights(x[i,t-1],sig_w)

        w[:,] /= sum(w[:,])    # normalize w   
    
        selected = np.random.choice(range(0,N),size=N,replace=True,p=w[:,t])
        x[:,t] = [sampleX(x[selected[i]][t],t,sig_v) for i in range(0,len(selected))]
        selected = np.random.choice(range(0,N),size=N,replace=True,p=np.repeat(1/N,N))

    return x , w

def SISR3Dplot(x,xLoLim=-25,xHiLim=25,lineCol='k',fillColMap='viridis',plotLines=False,stepSize=15,alpha=0,lineWidth=2,azim=245,elev=35.):
    # make 3d axes
    # stepSize can be max equal to T
    
    plt.clf()

    T = np.shape(x)[1]

    z = np.empty([T,2,100])
    y = np.empty([T,100])

    for t in range(0,T):    
        z[t] = sns.kdeplot(x[:,t], kernel='epa').get_lines()[0].get_data()
        y[t] = np.repeat(t,np.shape(z)[2])
        plt.clf()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    
    ax.set_xlim3d(xLoLim,xHiLim)
    ax.set_ylim3d(0,T)
    ax.set_zlim3d(0,np.max(z[:,1]))
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_xlabel('x_t')
    ax.set_ylabel('Time(t)')
    ax.set_zlabel('p(x_t | y_1:t')
    ax.view_init(azim=azim,elev=elev)
    cm = plt.get_cmap(fillColMap)
    cols = [cm(1. *i/(T-1)) for i in range(T)]

    for t in range(0,T,stepSize):
        cond = (z[t][0] <= xHiLim) & (z[t][0] >= xLoLim)
        tmpX = z[t][0][cond]
        tmpY = y[t][cond]
        tmpZ = z[t][1][cond]
        ax.add_collection3d(plt.fill_between(tmpX,0,tmpZ,alpha=alpha,color=cols[t],zorder=t+1),t,zdir='y') #color=fillCol
        if plotLines: ax.plot(tmpX,tmpY,tmpZ,lineWidth=lineWidth,color=cols[t],zorder=t+1)

    return fig

np.random.seed(910806)

mu_v = 0
var_v = 10
mu_w = 0
var_w = 1
priorMu = 0
priorVar = 10

N = 500
T = 200
 
x500 , w = SISR(mu_v,var_v,mu_w,var_w,priorMu,priorVar,N,T)

# plot results
thisplot500 = SISR3Dplot(x500,xLoLim=-25,xHiLim=25,lineCol='k',fillColMap='viridis',stepSize=18,alpha=0.7,lineWidth=1,azim=145,elev=35.)
plt.show()


'''
xEst = [np.dot(x[:,i],w[:,i]) for i in range(0,T)]
xTrue = np.zeros(T)
yTrue = np.zeros(T)
xTrue[0] = np.random.normal(0,np.sqrt(priorVar))

for t in range(1,T):
    xTrue[t] = 1/2*xTrue[t-1] + 25*xTrue[t-1]/(1 + xTrue[t-1]**2) + 8*math.cos(1.2*t) + np.random.normal(0,np.sqrt(var_v))
    yTrue[t] = xTrue[t]**2/20 + np.random.normal(xTrue[t],np.sqrt(var_w))

time = range(0,T)

yEst = [xEst[i]**2/20 + np.random.normal(xEst[i],np.sqrt(var_w)) for i in range(0,200)]

plt.plot(time,np.zeros(T),'k',linewidth=1)
plt.plot(time,x,'b',alpha=0.7,lineWidth=1)
plt.plot(time,xEst,'r',alpha=0.7,lineWidth=1)
plt.show()

'''