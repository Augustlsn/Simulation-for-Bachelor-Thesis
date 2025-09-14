import numpy as np
import sympy as sp
import array_to_latex as a2l 
import scipy.stats as sc


def vech(M):
    p = M.shape[0]
    return np.array([M[i,j] for j in range(p) for i in range(j,p)])
        
def generate_random_SigmaM(sparsity_mask = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 1]
    ], dtype=bool)  , diag_range=(-10,-5), offdiag_range=(-10, 10)):

    p = sparsity_mask.shape[0]
    assert p==2 or p==3
    
    M = np.zeros((p, p))
    Sigma_M = np.zeros((p,p))
    
    for i in range(p):
        for j in range(p):
            if i == j:
                M[i, j] = np.random.uniform(*diag_range)
            elif sparsity_mask[i, j]:
                M[i, j] = np.random.uniform(*offdiag_range)
                
    if(p==3):
        Sigma_M = np.array([
        [-1/(2*M[0,0]), M[1,0]/(2*M[0,0]*(M[0,0]+M[1,1])), -M[1,0]*M[2,1]/(2*M[0,0]*(M[0,0]+M[1,1])*(M[0,0]+M[2,2])) ],
        [M[1,0]/(2*M[0,0]*(M[0,0]+M[1,1])),-1*(M[1,0]**2+M[0,0]*(M[0,0]+M[1,1]))/(2*M[0,0]*M[1,1]*(M[0,0]+M[1,1])),((M[0,0]**2+M[1,0]**2)*(M[0,0]+M[1,1])*M[2,1]+(M[1,0]**2+M[0,0]*(M[0,0]+M[1,1]))*M[2,1]*M[2,2])/(2*M[0,0]*M[1,1]*(M[0,0]+M[1,1])*(M[0,0]+M[2,2])*(M[1,1]+M[2,2]))],
        [-M[1,0]*M[2,1]/(2*M[0,0]*(M[0,0]+M[1,1])*(M[0,0]+M[2,2])),((M[0,0]**2+M[1,0]**2)*(M[0,0]+M[1,1])*M[2,1]+(M[1,0]**2+M[0,0]*(M[0,0]+M[1,1]))*M[2,1]*M[2,2])/(2*M[0,0]*M[1,1]*(M[0,0]+M[1,1])*(M[0,0]+M[2,2])*(M[1,1]+M[2,2])), -1*(((M[0,0]+M[1,1])*(M[1,0]**2*M[2,1]**2+M[0,0]**2*(M[1,1]**2+M[2,1]**2))+(M[0,0]*M[1,1]*(M[0,0]+M[1,1])**2+(M[1,0]**2+M[0,0]*(M[0,0]+M[1,1]))*M[2,1]**2)*M[2,2]+M[0,0]*M[1,1]*(M[0,0]+M[1,1])*M[2,2]**2)/(2*M[0,0]*M[1,1]*(M[0,0]+M[1,1])*M[2,2]*(M[0,0]+M[2,2])*(M[1,1]+M[2,2]))) ]
        ])  

    if(p ==2):
        Sigma_M = np.array([
        [-1/(2*M[0,0]), M[1,0]/(2*M[0,0]*(M[0,0]+M[1,1]))],
        [M[1,0]/(2*M[0,0]*(M[0,0]+M[1,1])),-1*(M[1,0]**2+M[0,0]*(M[0,0]+M[1,1]))/(2*M[0,0]*M[1,1]*(M[0,0]+M[1,1]))]
        ]
        )
        
    return Sigma_M, M

def generate_SigmaM(M):
    p = M.shape[0]
    assert p ==2 or p==3


    if(p==3):
        Sigma_M = np.array([
        [-1/(2*M[0,0]), M[1,0]/(2*M[0,0]*(M[0,0]+M[1,1])), -M[1,0]*M[2,1]/(2*M[0,0]*(M[0,0]+M[1,1])*(M[0,0]+M[2,2])) ],
        [M[1,0]/(2*M[0,0]*(M[0,0]+M[1,1])),-1*(M[1,0]**2+M[0,0]*(M[0,0]+M[1,1]))/(2*M[0,0]*M[1,1]*(M[0,0]+M[1,1])),((M[0,0]**2+M[1,0]**2)*(M[0,0]+M[1,1])*M[2,1]+(M[1,0]**2+M[0,0]*(M[0,0]+M[1,1]))*M[2,1]*M[2,2])/(2*M[0,0]*M[1,1]*(M[0,0]+M[1,1])*(M[0,0]+M[2,2])*(M[1,1]+M[2,2]))],
        [-M[1,0]*M[2,1]/(2*M[0,0]*(M[0,0]+M[1,1])*(M[0,0]+M[2,2])),((M[0,0]**2+M[1,0]**2)*(M[0,0]+M[1,1])*M[2,1]+(M[1,0]**2+M[0,0]*(M[0,0]+M[1,1]))*M[2,1]*M[2,2])/(2*M[0,0]*M[1,1]*(M[0,0]+M[1,1])*(M[0,0]+M[2,2])*(M[1,1]+M[2,2])), -1*(((M[0,0]+M[1,1])*(M[1,0]**2*M[2,1]**2+M[0,0]**2*(M[1,1]**2+M[2,1]**2))+(M[0,0]*M[1,1]*(M[0,0]+M[1,1])**2+(M[1,0]**2+M[0,0]*(M[0,0]+M[1,1]))*M[2,1]**2)*M[2,2]+M[0,0]*M[1,1]*(M[0,0]+M[1,1])*M[2,2]**2)/(2*M[0,0]*M[1,1]*(M[0,0]+M[1,1])*M[2,2]*(M[0,0]+M[2,2])*(M[1,1]+M[2,2]))) ]
        ])  

    if(p ==2):
        Sigma_M = np.array([
        [-1/(2*M[0,0]), M[1,0]/(2*M[0,0]*(M[0,0]+M[1,1]))],
        [M[1,0]/(2*M[0,0]*(M[0,0]+M[1,1])),-1*(M[1,0]**2+M[0,0]*(M[0,0]+M[1,1]))/(2*M[0,0]*M[1,1]*(M[0,0]+M[1,1]))]
        ]
        )
        
    return Sigma_M, M

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

def fisher_information_M(M):
    p = M.shape[0]
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

    
    if(p == 3):
        theta = [m11,m21,m22,m32,m33]
        dSig = [sp.Matrix([[sp.diff(Sig_sy_3[i,j], th) for j in range(p)] for i in range(p)]) for th in theta]

        dSigma_M = sp.lambdify((m11,m21,m22,m32,m33), [sp.Matrix(d) for d in dSig], 'numpy')    

        dS = dSigma_M(M[0,0], M[1,0], M[1,1], M[2,1], M[2,2])

        Sigm_builder = sp.lambdify((m11,m21,m22,m32,m33),Sig_sy_3,'numpy')
        Sigma = Sigm_builder(M[0,0], M[1,0], M[1,1], M[2,1], M[2,2])
        Sigma_inv = np.linalg.inv(Sigma)
        Fisher = np.zeros((5,5))

        for a in range(5):
            for b in range(5):
                Fisher[a,b] = 1/2 * np.trace(Sigma_inv @ np.array(dS[a], dtype = float) @ Sigma_inv @ np.array(dS[b],dtype = float))

    
    if(p==2): 
        theta = [m11,m21,m22]
        dSig = [sp.Matrix([[sp.diff(Sig_sy_2[i,j], th) for j in range(p)] for i in range(p)]) for th in theta]

        dSigma_M = sp.lambdify((m11,m21,m22), [sp.Matrix(d) for d in dSig], 'numpy')    
        Sigm_builder = sp.lambdify((m11,m21,m22),Sig_sy_2,'numpy')
        Sigma = Sigm_builder(M[0,0], M[1,0], M[1,1])
        Sigma_inv = np.linalg.inv(Sigma)
        dS = dSigma_M(M[0,0], M[1,0], M[1,1])
       
    
        Fisher = np.zeros((3,3))

        for a in range(3):
            for b in range(3):
                Fisher[a,b] = 1/2 * np.trace(Sigma_inv @ np.array(dS[a], dtype = float) @ Sigma_inv @ np.array(dS[b],dtype = float))

    return Fisher

def simulate_asymptvariance(Sigma,M,nsample,niter):
    p = Sigma.shape[0]
    assert p == 2 or p==3

    fisher_inf = fisher_information_M(M)
    I = -1 * vech(np.eye(p))
    
    def transform_g(sigma, Matrix: bool):
        if(Matrix):
            A = construct_A_masked(vech(sigma))
            return np.linalg.lstsq(A,I,rcond=1e-15)[0]
        else:
            A = construct_A_masked(sigma)
            return np.linalg.lstsq(A,I,rcond=1e-15)[0]
                 
    def dA_masked_dsigma(i):
        if(p==3):
            dA = np.zeros((6, 5), dtype= float)
            if i == 0:  # s11
                dA[0, 0] = 2
                dA[1, 1] = 1
            elif i == 1:  # s21
                dA[1, 2] = 1
                dA[1, 0] = 1
                dA[3, 1] = 2
                dA[2, 3] = 1
            elif i == 2:  # s31
                dA[2, 0] = 1
                dA[2, 4] = 1
                dA[4, 1] = 1
            elif i == 3:  # s22
                dA[3, 2] = 2
                dA[4, 3] = 1
            elif i == 4:  # s32
                dA[4, 2] = 1
                dA[5,3] = 2
                dA[4,4] = 1
            elif i == 5:  # s33
                dA[5, 4] = 2

        if(p==2):
            dA = np.zeros((3,3), dtype=float)

            if i == 0:  # s11
                dA[0, 0] = 2
                dA[1, 1] = 1
            elif i == 1:  # s21
                dA[1, 2] = 1
                dA[1, 0] = 1
                dA[2, 1] = 2
            elif i == 2: #s22
                dA[2, 2] = 2
 
        return dA
    
    def asympt_cov(Sigma):
        if(p==3):
            vech_idx = [(0,0),(1,0),(2,0),(1,1),(2,1),(2,2)]
            V = np.empty((6,6), dtype=float)

        if(p==2):
            vech_idx = [(0,0),(1,0),(1,1)]
            V = np.empty((3,3), dtype= float)

        for a, (i,j) in enumerate(vech_idx):
            for b, (k,l) in enumerate(vech_idx):
                V[a,b] = Sigma[i,k]*Sigma[j,l] + Sigma[i,l]*Sigma[j,k]

        return V
    
    def calculate_derivatives_by_sigma(Sigma):
        A = construct_A_masked(vech(Sigma))
        A_T = A.T 
        
        if(p==3):
            A_inv = np.linalg.pinv(A_T @ A, rcond= 1e-10)
            Jakobi = np.zeros((5, 6), dtype=float)
            for k in range(6):
                dA = dA_masked_dsigma(k)
                dA_T = dA.T 
                dA_final = dA_T @ A + A_T @ dA

                term1 = -A_inv @ dA_final @ A_inv @ A_T
                term2 = A_inv @ dA_T

                Jakobi[:, k] = (term1 + term2) @ I

        if(p==2):
            A_inv = np.linalg.inv(A)
            Jakobi = np.zeros((3, 3), dtype=float)

            for k in range(3):
                dA = dA_masked_dsigma(k)
                term1 = -1 * A_inv @ dA @ A_inv 
                Jakobi[:, k] = term1 @ I

        return Jakobi
    
    J = calculate_derivatives_by_sigma(Sigma)
    V = asympt_cov(Sigma)
  

    if(p==3):
        diffs = np.empty((niter,5),float)
        M0 = np.array([M[0,0],M[1,0],M[1,1],M[2,1],M[2,2]])
    if(p==2):
        diffs = np.empty((niter,3),dtype = float)    
        M0 = np.array([M[0,0],M[1,0],M[1,1]])
    
    for i in range(niter):
        S_hat = sc.wishart.rvs(df = nsample, scale = Sigma)/nsample
        m_hat = transform_g(S_hat, Matrix=True)
        
        diffs[i] = np.sqrt(nsample)*(m_hat - M0)
        
    vartheo = J @ V @ J.T
    varest = np.cov(diffs,rowvar=False)
    

    return varest, vartheo, fisher_inf
    

if __name__ == "__main__":
    #Example
    np.random.seed(13)
    M = np.array([[-1.5, 0, 0],
                  [.7,-1.1, 0],
                  [0,0.25,-1.8]
                  ])
    
    Sigma = generate_SigmaM(M)
   
    var_est, var_theo, fisher = simulate_asymptvariance(Sigma, M, nsample = 10000, niter = 100000)
    fisher_inv = np.linalg.inv(fisher) 
   

    

    
    
    
    
    
    
    


    
    
    

