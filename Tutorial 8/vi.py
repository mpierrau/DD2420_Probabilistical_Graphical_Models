# author Olga Mikheeva olgamik@kth.se
# PGM tutorial on Variational Inference
# Bayesian Mixture of Gaussians

import numpy as np
import matplotlib.pyplot as plt
import math


def generate_data(std, k, n, dim=1):
    means = np.random.normal(0.0, std, size=(k, dim))
    data = []
    categories = []
    for i in range(n):
        cat = np.random.choice(k)  # sample component assignment
        categories.append(cat)
        data.append(np.random.multivariate_normal(means[cat, :], np.eye(dim)))  # sample data point from the Gaussian, lambda=1
    return np.stack(data), categories, means


def plot(x, y, c, means, title):
    plt.scatter(x, y, c=c)
    plt.scatter(means[:, 0], means[:, 1], c='r')
    plt.title(title)
    plt.show()


def plot_elbo(elbo):
    plt.plot(elbo)
    plt.title('ELBO')
    plt.show()


def compute_elbo(data, psi, m, s2, sigma2, mu0):
    """ Computes ELBO
    The outcommented codeparts contain expressions that are constants and are not 
    needed to compute ELBO as a comparative measure
    """

    n, p = data.shape
    k = m.shape[0]

    # compute ELBO
    # expected log prior over mixture assignments
    #Epci = -n*np.log(k)
    Epci = 0

    # expected log prior over mixture locations
    #muTerm1 = p*np.sum(np.log(2*math.pi*s2),axis=(0,1)) - k/2
    #muTerm2 = p*np.sum(s2,axis=(0))
    #muTerm3 = np.sum((m-mu0.T)**2,axis=(0,1))


    muTerm1 = np.sum(np.log(s2),axis=(0))
    muTerm2 = np.sum(s2,axis=(0))
    muTerm3 = np.sum((m-mu0.T)**2,axis=(0,1))

    Epmu = -muTerm1/2 - (muTerm2 + muTerm3)/(2*sigma2)

    # expected log likelihood
    #LHTerm1 = p/2*np.log(2*math.pi)*np.sum(psi,axis=(0,1))
    #LHTerm2 = p*np.dot(np.sum(psi,0),s2)
    #LHConsTerm = np.dot(np.sum(psi,1),np.sum(data*data,1)) #This term will be constant given data since Psi sums to 1 over k


    LHTerm1 = np.sum(psi,axis=(0,1))
    LHTerm2 = np.dot(np.sum(psi,0),s2)
    LHLinTerm = np.sum(np.tensordot(data,m,axes=[[1],[1]])*psi)
    LHQuadTerm = np.dot(np.sum(m*m,1),np.sum(psi,0))

    ELH = -LHTerm1-LHTerm2+2*LHLinTerm-LHQuadTerm

    # entropy of variational location posterior

    #Eqmu = -p/2*(np.sum(np.log(2*math.pi*s2)) + k)
    Eqmu = np.sum(np.log(s2))

    # entropy of the variational assignment posterior

    Eqci = np.sum(psi*np.log(psi),axis=(0,1))

    elbo = Epci + Epmu + ELH - Eqmu - Eqci
    return elbo


def cavi(data, k, sigma2, m0, eps=1e-15):
    """ Coordinate ascent Variational Inference for Bayesian Mixture of Gaussians
    :param data: data
    :param k: number of components
    :param sigma2: prior variance
    :param m0: prior mean
    :param eps: stopping condition
    :return (m_k, s2_k, psi_i)
    """
    n, p = data.shape
    # initialize randomly
    m = np.random.normal(0., 1., size=(k, p))
    s2 = np.square(np.random.normal(0., 1., size=(k, 1)))
    psi = np.random.dirichlet(np.ones(k), size=n)

    # compute ELBO
    elbo = [compute_elbo(data, psi, m, s2, sigma2, m0)]
    convergence = 1.
    iter = 1
    while convergence > eps:  # while ELBO not converged
        # update categorical

        Emu = m
        EmuSq = m**2 + np.reshape(s2**2,(5,1))

        expPart1 = np.tensordot(data,Emu,axes=[[1],[1]])
        expPart2 = np.sum(m**2,1)+s2.T
        psi = np.exp(expPart1-expPart2/2)
        psi /= np.sum(psi,1)[:,None] # Normalizing
        
        # update posterior parameters for the component means

        s2 = sigma2/(1 + sigma2*np.sum(psi,0))
        m = (s2*np.tensordot(psi,data,axes=[[0],[0]]).T).T

        # compute ELBO
        elbo.append(compute_elbo(data, psi, m, s2, sigma2, m0))
        
        convergence = elbo[-1] - elbo[-2]

    return m, s2, psi, elbo


def main():
    np.random.seed(19910800)
    #np.random.seed(19910800) for overlapping clusters

    # parameters
    p = 2
    k = 5
    sigma = 5.

    data, categories, means = generate_data(std=sigma, k=k, n=500, dim=p)
    m = list()
    s2 = list()
    psi = list()
    elbo = list()
    best_i = 0
    for i in range(10):
        m_i, s2_i, psi_i, elbo_i = cavi(data, k=k, sigma2=sigma, m0=np.zeros(p))
        m.append(m_i)
        s2.append(s2_i)
        psi.append(psi_i)
        elbo.append(elbo_i)
        if i > 0 and elbo[-1][-1] > elbo[best_i][-1]:
            best_i = i
    class_pred = np.argmax(psi[best_i], axis=1)
    plot(data[:, 0], data[:, 1], categories, means, title='true data')
    plot(data[:, 0], data[:, 1], class_pred, m[best_i], title='posterior')
    plot_elbo(elbo[best_i])
    print(s2[best_i])
    print((np.sum(psi,0)/np.sum(psi,axis=(0,1)))[best_i])

if __name__ == '__main__':
    main()
