#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# The file is based on AQAE.py of https://github.com/randylewis/QuantumAnnealing
# goal: comparing different fixed-point approx. (arxiv: 2103.08661 and 2202.12340)

from collections  import defaultdict
from numpy        import sqrt, array, identity, tril_indices, matmul, diag, fill_diagonal, copy
from dwave.system import DWaveSampler, EmbeddingComposite
from neal         import SimulatedAnnealingSampler

# BEGIN USER INPUTS.
center_zoom = 0 # Centering zoom as in 2202.12340 (1) or 2103.08661 (0)
quantum = 1     # Choose 0 for SimulatedAnnealingSampler. Choose 1 for DWaveSampler.
myx = 0.6       # Choose the value of the parameter x that appears in the Hamiltonian matrix.
myK = 4         # Choose the initial number of qubits used for each row of the Hamiltonian matrix.
mylambda = -1.9 # Choose the coefficient of the constraint for the QAE algorithm.
myreads = 1000 # Choose the number of samples to be used by the D-Wave sampler.
mychain = 3.0   # Choose a value for the chain penalty.
mytime = 20     # Choose the annealing time in microseconds (integer between 1 and 2000 inclusive).
# END USER INPUTS


# Define the number of rows in the Hamiltonian matrix.
myB = 13

# Define the Hamiltonian, including the constraint with coefficient lambda.
Hlambda = defaultdict(float)
Hlambda[(1,1)] = 0.0 - mylambda
Hlambda[(1,2)] = -2.0*sqrt(6.0)*myx
Hlambda[(2,2)] = 3.0 - mylambda
Hlambda[(2,3)] = -2.0*myx
Hlambda[(2,4)] = -4.0*myx
Hlambda[(2,5)] = -2.0*sqrt(2.0)*myx
Hlambda[(3,3)] = 4.5 - mylambda
Hlambda[(3,6)] = -2.0*myx
Hlambda[(3,7)] = -2.0*sqrt(2.0)*myx
Hlambda[(4,4)] = 6.0 - mylambda
Hlambda[(4,6)] = -myx/2.0
Hlambda[(4,7)] = -sqrt(2.0)*myx
Hlambda[(4,8)] = -2.0*sqrt(3.0)*myx
Hlambda[(5,5)] = 6.0 - mylambda
Hlambda[(5,7)] = -2.0*myx
Hlambda[(6,6)] = 6.0 - mylambda
Hlambda[(6,9)] = -2.0*myx
Hlambda[(6,10)] = -2.0*myx
Hlambda[(7,7)] = 7.5 - mylambda
Hlambda[(7,9)] = -myx/sqrt(2.0)
Hlambda[(7,10)] = -sqrt(2.0)*myx
Hlambda[(7,11)] = -2.0*myx
Hlambda[(8,8)] = 9.0 - mylambda
Hlambda[(8,10)] = -sqrt(3.0)*myx/2.0
Hlambda[(9,9)] = 7.5 - mylambda
Hlambda[(9,12)] = -2.0*myx
Hlambda[(10,10)] = 9.0 - mylambda
Hlambda[(10,12)] = -myx
Hlambda[(11,11)] = 9.0 - mylambda
Hlambda[(11,12)] = -myx/sqrt(2.0)
Hlambda[(12,12)] = 9.0 - mylambda
Hlambda[(12,13)] = -sqrt(1.5)*myx
Hlambda[(13,13)] = 9.0 - mylambda

# Initialize the center location of the solution vector.
acenter = [0]*myB

# Define the zoom factor (number of extra factors of 2) of the matrix.
myz = 0

if center_zoom==1:
    hlambda = defaultdict(float)
    hlambda = Hlambda
    for alpha in range(1,myB+1):
            for beta in range(alpha+1,myB+1):
                if (alpha,beta) in Hlambda:
                    hlambda[beta, alpha] = Hlambda[alpha, beta]

# Repeat the computation with a finer discretization until the user terminates the loop.
while True:

# Define the matrix Q of the AQAE algorithm.
    Q = defaultdict(float)
    if(center_zoom==0):
        for alpha in range(1,myB+1):
            for beta in range(alpha,myB+1):
                if (alpha,beta) in Hlambda:
                    for n in range(1,myK+1):
                        i = myK*(alpha-1) + n
                        for m in range(1,myK+1):
                            j = myK*(beta-1) + m
                            if i<=j:
                                if n==myK and m==myK:
                                    Q[i,j] = (acenter[alpha-1]-2**(-myz))*(acenter[beta-1]-2**(-myz))*Hlambda[alpha,beta]
                                elif n==myK:
                                    Q[i,j] = (acenter[alpha-1]-2**(-myz))*2**(m-myK-myz)*Hlambda[alpha,beta]
                                elif m==myK:
                                    Q[i,j] = (acenter[beta-1]-2**(-myz))*2**(n-myK-myz)*Hlambda[alpha,beta]
                                else:
                                    Q[i,j] = 2**(n+m-2*myK-2*myz)*Hlambda[alpha,beta]
                            if i<j:
                                Q[i,j] *= 2
    else:                         
        for alpha in range(1,myB+1):
            for beta in range(alpha,myB+1):
                if (alpha,beta) in Hlambda:
                    for i in range(1,myK+1):
                        n = myK*(alpha-1) + i
                        for j in range(1,myK+1):
                            m = myK*(beta-1) + j
                            if n<=m:
                                prefac = 1;
                                if i==myK:
                                    prefac=-1
                                if j==myK:
                                    prefac*=-1
                                Q[n,m] = 2**(i+j-2*myK-2*myz)*prefac*Hlambda[alpha,beta]
                            if n<m:
                                Q[n,m] *= 2
      
        for alpha in range(1,myB+1):
            for i in range(1,myK+1):
                n = myK*(alpha-1) + i
                prefac = 1;
                if i==myK:
                    prefac=-1
                summa = 0
                for gamma in range(1,myB+1):
                    if (gamma,alpha) in hlambda:
                        summa += acenter[gamma-1]*hlambda[gamma, alpha]
                Q[n,n] += 2**(i-myK-myz+1)*prefac*summa

# Send the job to the requested sampler.
    if quantum==1:
        print("Using DWaveSampler")
        sampler = EmbeddingComposite(DWaveSampler(solver={'topology__type__eq':'pegasus'}))
        sampleset = sampler.sample_qubo(Q,num_reads=myreads,chain_strength=mychain,annealing_time=mytime)
        rawoutput = sampleset.aggregate()
    else:
        print("Using SimulatedAnnealingSampler")
        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample_qubo(Q,num_reads=myreads)
        rawoutput = sampleset.aggregate()

# Translate the vectors from the Q basis to the H basis, and then display the final results.
    minimumevalue = 100.0
    warning = 0
    chaincount = 0
    for irow in range(len(rawoutput.record)):
        if quantum==1:
            chain = rawoutput.record[irow][3]
        numoc = rawoutput.record[irow][2]
        a = []
        for alphaminus1 in range(myB):
            a.append(0)
            for kminus1 in range(myK-1):
                i = myK*alphaminus1 + kminus1
                a[alphaminus1] += 2**(1+kminus1-myK-myz)*rawoutput.record[irow][0][i]
            i = myK*alphaminus1 + myK - 1
            if center_zoom==1:
                a[alphaminus1] += acenter[alphaminus1]-2**(-myz)*rawoutput.record[irow][0][i]
            else:
                a[alphaminus1] += (acenter[alphaminus1]-2**(-myz))*rawoutput.record[irow][0][i]
        anorm = sqrt(sum(a[i]**2 for i in range(myB)))
        if anorm<1.0e-6:
            print('{:7d}'.format(numoc), "   ---    This vector has length zero.")
            warning += numoc
        else:
            evalue = mylambda + rawoutput.record[irow][1]/anorm**2
            unita = [a[i]/anorm for i in range(myB)]
            if (quantum==1):
                print('{:07.6f}'.format(chain), end =" ")
                if chain>1.0e-6:
                    chaincount += 1
            #print('{:7d}'.format(numoc), '{:07.6f}'.format(evalue), ' '.join('{: 07.6f}'.format(f) for f in unita))
            minimumevalue = min(evalue,minimumevalue)
            if evalue==minimumevalue:
                minimuma = a
                minimumunita = unita
                if quantum==1:
                    minimumchain = chain
    #print("The minimum evalue from the physics output above is ",'{:07.6f}'.format(minimumevalue)) -> missing the constant term in Fs
    print("The normalized evector for the minimum evalue is ",' '.join('{: 07.6f}'.format(f) for f in minimumunita))
    if (quantum==1):
        print("The chain breaking for the minimum evalue is ",'{:07.6f}'.format(minimumchain))
        print("The number of reads that have broken chains is ",chaincount)
    if (warning>0):
        print("WARNING: The number of reads giving the vector of length zero is",warning)

# Construct H_E, H_B, and the full symmetric Hamiltonian matrix without lambda.
    temp_x, temp_y = map(max, zip(*Hlambda)) 
    myH = array([[Hlambda.get((j, i), 0) for i in range(1,temp_y + 1)] for j in range(1,temp_x + 1)])
    myH = myH + mylambda*identity(myB,dtype=float)
    i_lower = tril_indices(myB, -1)
    myH[i_lower] = myH.T[i_lower]
    myHE = diag(diag(myH))
    myHB = copy(myH)
    fill_diagonal(myHB,0.0)

# Calculate <H_E> and <H_B> and <H> for the ground state.
    HEave = matmul(minimumunita,matmul(myHE,minimumunita))
    HBave = matmul(minimumunita,matmul(myHB,minimumunita))
    Have = matmul(minimumunita,matmul(myH,minimumunita))
    print("The normalized evector gives <H_E>, <H_B>, <H> =", HEave, HBave, Have)

# Ask the user whether another computation should be done.
    print()
    print("This has been a computation with myK+zoom =",myK,"+",myz,"=",myK+myz)
    query="yes"
    #query = input("Do you want to try a more precise computation based on this starting point? (yes/no/y/n) ") 
    #print("User input=",query)
    if query=="no" or query=="n":
        print()
        print("===============================================================")
        print()
        break

# Prepare for the next computation.
    myz += 1
    acenter = minimuma
    print()
    print("===============================================================")
    print()

