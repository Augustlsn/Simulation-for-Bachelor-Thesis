import numpy as np
import scipy.stats as sc
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import sympy as sp

def vech(M):
    p = M.shape[0]
    return np.array([M[i,j] for j in range(p) for i in range(j,p)])

def construct_A_masked(sigma):
    p = sigma.shape[0]
    assert p == 6 or p==3
    if(p == 6):
        s11, s21, s31, s22, s32, s33 = sigma
        A = np.array([
        [2*s11, 0,      0,      0,      0],  
        [s21,   s11,     s21,    0,      0],  
        [s31,   0,       0,    s21,   s31],   
        [0,       2*s21,  2*s22,  0,      0],   
        [0,       s31,   s32,   s22,   s32],      
        [0,       0,      0,   2*s32, 2*s33]     
        ])
    if(p==3):
        s11, s21, s22= sigma
        A = np.array([
        [2*s11, 0, 0],  
        [s21, s11, s21],    
        [0, 2*s21, 2*s22]   
        ])
    return A

def generate_Sigma_M(m11,m21,m22,m32 = 0,m33 = 0):
    if(m33 == 0):
        S11 = -1/(2*m11)
        S21 =  m21/(2*m11*(m11+m22))
        S22 = -((m21)**2+m11*(m11+m22))/(2*m11*m22*(m11+m22))
        
        Sigma = np.array([
            [S11,S21],
            [S21,S22]
        ])
        return Sigma
    
    else:    
        S11 = -1/(2*m11)
        S21 =  m21/(2*m11*(m11+m22))
        S31 = -m21*m32/(2*m11*(m11+m22)*(m11+m33))
        S22 = -((m21)**2+m11*(m11+m22))/(2*m11*m22*(m11+m22))
        S32 = ((m11**2+m21**2)*(m11+m22)*m32+(m21**2+m11*(m11+m22))*m32*m33)/(2*m11*m22*(m11+m22)*(m11+m33)*(m22+m33))
        S33 = -((m11+m22)*(m21**2*m32**2+m11**2*(m22**2+m32**2))+(m11*m22*(m11+m22)**2+(m21**2+m11*(m11+m22))*m32**2)*m33+m11*m22*(m11+m22)*m33**2)/(2*m11*m22*(m11+m22)*m33*(m11+m33)*(m22+m33))

        Sigma = np.array([
            [S11, S21, S31],
            [S21, S22, S32],
            [S31, S32, S33]
        ])
        return Sigma
   
def data_simulation(Sigma,nsample):
    S_hat = sc.wishart.rvs(df = nsample, scale = Sigma) /nsample
    return S_hat

def log_posterior(m, S_hat, nsample):
    p = m.shape[0]

    if(p==3):
        m11,m21,m22 = m

        if(m11 >= 0 or m22 >= 0):
            return -np.inf

        try:
            Sigma = generate_Sigma_M(m11,m21,m22)
            
            sign,logdet = np.linalg.slogdet(Sigma)
            if sign != 1:
                return -np.inf
            S_inv = np.linalg.inv(Sigma)
        except:
            return -np.inf
        
        tracet = np.trace(S_inv @ S_hat)
        loglike = -.5 *nsample*logdet - .5*nsample*tracet
        logprior = m11 + m22 - .5*m21**2
        
    if(p==5):
        m11,m21,m22,m32,m33 = m

        if(m11 >= 0 or m22 >= 0 or m33>= 0):
            return -np.inf
        try:
            Sigma = generate_Sigma_M(m11,m21,m22,m32,m33)
            
            sign,logdet = np.linalg.slogdet(Sigma)
            if(sign != 1):
                return -np.inf
            S_inv = np.linalg.inv(Sigma)
        except:
            return -np.inf
        
        tracet = np.trace(S_inv @ S_hat)
        loglike = -.5 * nsample * logdet - .5 * nsample * tracet
        logprior = m11 + m22 + m33 - .5 * (m21**2 + m32 **2)
        
    return loglike + logprior

def autocorr(data,lag,k):
    '''
    refernce: https://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation
    '''
    corr = [1. if l ==0 else np.corrcoef(data[l:,k],data[:-l,k])[0][1] for l in lag]
    return np.array(corr)

def metropolis_sampler(init,S_hat,nsample,ngenerate,sigma,burn_in,lag):
    p=init.shape[0]
    iterationnumber = lag*ngenerate+burn_in #burnin + autocorrelation
    samples = []
    theta = np.array(init)  
    lp = log_posterior(theta,S_hat,nsample)  
    k = 0
    for _ in range(iterationnumber):
        prop = theta + np.random.normal(0,sigma,size = p)
        lp_prop = log_posterior(prop,S_hat,nsample)
        
        if(np.log(np.random.uniform()) <= (lp_prop - lp)):
            theta = prop
            lp = lp_prop
            k +=1

        samples.append(theta.copy())    
    #print(k/iterationnumber)
    sampl_fin = np.array(samples[burn_in::lag])
    
    return sampl_fin  

def fisher_information_M(Sigma,M):
    p = Sigma.shape[0]
    assert p == 3 or p==2
    
    
    m11, m21, m22, m32, m33 = sp.symbols('m11 m21 m22 m32 m33', real=True)
    
    S11 = -1/(2*m11)
    S12 =  m21/(2*m11*(m11+m22))
    S13 = -m21*m32/(2*m11*(m11+m22)*(m11+m33))
    S22 = -((m21)**2+m11*(m11+m22))/(2*m11*m22*(m11+m22))
    S23 = ((m11**2+m21**2)*(m11+m22)*m32+(m21**2+m11*(m11+m22))*m32*m33)/(2*m11*m22*(m11+m22)*(m11+m33)*(m22+m33))
    S33 = -((m11+m22)*(m21**2*m32**2+m11**2*(m22**2+m32**2))+(m11*m22*(m11+m22)**2+(m21**2+m11*(m11+m22))*m32**2)*m33+m11*m22*(m11+m22)*m33**2)/(2*m11*m22*(m11+m22)*m33*(m11+m33)*(m22+m33))

    Sig_sy_3 = sp.Matrix([[S11,S12,S13],[S12,S22,S23],[S13,S23,S33]])
    Sig_sy_2 = sp.Matrix([[S11,S12],[S12,S22]])

    # derivative tensors: 5 x 3 x 3 
    if(p == 3):
        theta = [m11,m21,m22,m32,m33]
        dSig = [sp.Matrix([[sp.diff(Sig_sy_3[i,j], th) for j in range(p)] for i in range(p)]) for th in theta]

        dSigma_M = sp.lambdify((m11,m21,m22,m32,m33), [sp.Matrix(d) for d in dSig], 'numpy')    

        dS = dSigma_M(M[0,0], M[1,0], M[1,1], M[2,1], M[2,2])
        Sigma_inv = np.linalg.inv(Sigma)

        Fisher = np.zeros((5,5))

        for a in range(5):
            for b in range(5):
                Fisher[a,b] = 1/2 * np.trace(Sigma_inv @ np.array(dS[a], dtype = float) @ Sigma_inv @ np.array(dS[b],dtype = float))
    if(p==2): 
        theta = [m11,m21,m22]
        dSig = [sp.Matrix([[sp.diff(Sig_sy_2[i,j], th) for j in range(p)] for i in range(p)]) for th in theta]

        dSigma_M = sp.lambdify((m11,m21,m22), [sp.Matrix(d) for d in dSig], 'numpy')    

        dS = dSigma_M(M[0,0], M[1,0], M[1,1])
        Sigma_inv = np.linalg.inv(Sigma)
    
        Fisher = np.zeros((3,3))

        for a in range(3):
            for b in range(3):
                Fisher[a,b] = 1/2 * np.trace(Sigma_inv @ np.array(dS[a], dtype = float) @ Sigma_inv @ np.array(dS[b],dtype = float))

    return Fisher

def estimator(S_hat):
    p = S_hat.shape[0]
    
    I = -vech(np.eye(p))
    A = construct_A_masked(vech(S_hat))
    return np.linalg.lstsq(A,I,rcond=1e-10)[0]

def Simulation_and_plot(m,nsample,ngenerate,sigma,burn_in,init,plotting: bool): #sampling from the posterior
    p = m.shape[0]

    if(p==3):
        m11,m21,m22 = m
        Sigma = generate_Sigma_M(m11,m21,m22)
        M = np.array([[m11,0],[m21,m22]])
        fisher = fisher_information_M(Sigma,M)
        fisher_inv = np.linalg.inv(fisher)

    if(p==5):
        m11,m21,m22,m32,m33 = m
        Sigma = generate_Sigma_M(m11,m21,m22,m32,m33)
        M = np.array([[m11,0,0],[m21,m22,0],[0,m32,m33]])
        fisher = fisher_information_M(Sigma,M)
        fisher_inv = np.linalg.inv(fisher)


    S_hat = data_simulation(Sigma,nsample)
    samples = metropolis_sampler(init,S_hat,nsample,ngenerate,sigma,burn_in,5)
    

    if(p==3):
        params = ["m11","m21","m22"]
        samples_cent = np.sqrt(nsample)*(samples-np.tile(estimator(S_hat),(ngenerate,1)))
    if(p==5):
        params = ["m11","m21","m22","m32","m33"]  
        samples_cent = np.sqrt(nsample)*(samples-np.tile(samples.mean(axis = 0),(ngenerate,1)))  
    
    if(plotting):
        fig,axes = plt.subplots(2,len(params),figsize =(4*len(params),9),sharex = 'col',sharey='row')
        axes = np.atleast_2d(axes)
        for row,param in enumerate(params):
            sd =  np.sqrt(fisher_inv[row,row])
            data = samples_cent[:,row]
            
          
            ax_q = axes[1, row]

            theo_q, samp_q = sc.probplot(data/sd, dist=norm, fit = False)

            ax_q.scatter(theo_q, samp_q, s=14, alpha=0.75)
            lo, hi = np.min(theo_q), np.max(theo_q)
            ax_q.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
            ax_q.set_ylabel(param)
            ax_q.legend(loc="lower right", fontsize=8)


            
            ax_h = axes[0, row]
            ax_h.hist(data,bins = 50, density = True, alpha = .6, label = "posterior")
            ax_h.set_xlim(-5*sd,+5*sd)
            x_vals = np.linspace(-5*sd,+5*sd,100)
            ax_h.plot(x_vals, norm.pdf(x_vals,0, sd),"k--",label = "N(0,1)")
            ax_h.set_ylabel(param)

            if row ==0:
                ax_h.set_ylabel("Density")
                ax_q.set_ylabel("Sample quantiles")
            
        for ax in np.ravel(axes[1,:]):
            ax.set_xlabel("Theoretical quantiles N(0,1)")
        fig.tight_layout()
        plt.show()

def asymptotic_var_sim(m,nsample,ngenerate,nvar,sigma,burn_in,init,plotting): #sample variance of the posterior mean
    p = m.shape[0]
    assert p ==3 or p ==5

    if(p==3):
        params = ["m11","m21","m22"]
        m11,m21,m22 = m
        Sigma = generate_Sigma_M(m11,m21,m22)
        M = np.array([[m11,0],[m21,m22]])
        fisher = fisher_information_M(Sigma,M)
        fisher_inv = np.linalg.inv(fisher)

    if(p==5):
        params = ["m11","m21","m22","m32","m33"] 
        m11,m21,m22,m32,m33 = m
        Sigma = generate_Sigma_M(m11,m21,m22,m32,m33)
        M = np.array([[m11,0,0],[m21,m22,0],[0,m32,m33]])
        fisher = fisher_information_M(Sigma,M)
        fisher_inv = np.linalg.inv(fisher)
        

    theta = np.empty((nvar,p),dtype=float)
    
    for i in range(nvar):
        S_hat = data_simulation(Sigma,nsample)
        samples = metropolis_sampler(init,S_hat,nsample,ngenerate,sigma,burn_in,5)
        theta[i,:] = samples.mean(axis = 0)
        
    
    var_est = np.cov(np.sqrt(nsample)*(theta-np.tile(m,(nvar,1))),rowvar = False)
    data_cent = np.sqrt(nsample)*(theta-np.tile(m,(nvar,1)))

    if(plotting):
        fig,axes = plt.subplots(2,len(params),figsize =(6*len(params),8),sharex = 'col',sharey='row')
        axes = np.atleast_2d(axes)

        for col,param in enumerate(params):
            sd = np.sqrt(fisher_inv[col,col])
            data_stand = data_cent[:,col] / sd

            
            ax_q = axes[1, col]

            theo_q, samp_q = sc.probplot(data_stand, dist=norm, fit = False)

            ax_q.scatter(theo_q, samp_q, s=14, alpha=0.75)
            lo, hi = np.min(theo_q), np.max(theo_q)
            ax_q.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
            ax_q.set_ylabel(param)
            ax_q.legend(loc="lower right", fontsize=8)


            ax_h = axes[0, col]
            ax_h.hist(sd*data_stand,bins = 100, density = True, alpha = .6, label = "posterior")
            ax_h.set_xlim(-4*sd,+4*sd)
            x_vals = np.linspace(-4*sd,+4*sd,100)
            ax_h.plot(x_vals, norm.pdf(x_vals,0, np.sqrt(fisher_inv[col,col])),"k--",label = "N(0,1)")
            ax_h.set_ylabel(param)

            if col ==0:
                ax_h.set_ylabel("Density")
                ax_q.set_ylabel("Sample quantiles")
            
        for ax in np.ravel(axes[1,:]):
            ax.set_xlabel("Theoretical quantiles N(0,1)")

        fig.tight_layout()
        plt.show()

    return var_est,fisher_inv
       
if __name__ == "__main__":
    #Example
    np.random.seed(15)
   
    m = np.array([-1.5,-.3,-1.1])
    nsample = 10000
    ngenerate = 1000
    sigma = 0.025
    burn_in = 100
    init = -np.array([1,0,1])
 
    var_est, fisher_inv = asymptotic_var_sim(m,nsample,ngenerate,100,sigma,burn_in,init,True)
   