from math import *
from random import *

def matr_vec_mult(A, b):
    C = list()
    for i in range(len(A)):
        C.append(0.)
        for j in range(len(A[i])):
            C[i] += A[i][j]*b[j]
    return C
#=================================
# input specifics
#=================================
T = 2.5
theta = 0.5
N = 5
L = 3
delays = [3, 5, 6]
inp = [1.,1.,1.,1.]
outsize = 2
#=================================
#initial generation
#=================================
def gen_M(N, L, delays):
    M = list()
    for i in range(N*L):
        M.append([0.] * len(delays))
        for j in range(len(delays)):
            if (i > N-1) and (i%N - delays[j]) < 0 and (i%N - delays[j]) > -N - 1:
                M[i][j] = uniform(-10.,10.)
    return M

M = gen_M(N, L, delays)

def gen_Win(N, inp):
    Win = list()
    for i in range(N):
        Win.append(list())
        for j in range(len(inp) + 1):
            Win[i].append(uniform(-10.,10.))
    return Win

Win = gen_Win(N, inp)

def gen_Wout(N, outsize):
    Wout = list()
    for i in range(outsize):
        Wout.append(list())
        for j in range(N+1):
            Wout[i].append(uniform(-10.,10.))
    return Wout

Wout = gen_Wout(N, 2)

def gen_z(L, N):
    z = [None] * (L*N)
    for i in range(len(z)):
        z[i] = 10*cos(5*i - 6)
    return z

z = gen_z(L, N)
#===========================================
#input preprocessing
#===========================================
def J(Win, inp):
    WinU = matr_vec_mult(Win, inp + [1.])
    for i in range(len (WinU)):
        WinU[i] = f_in(WinU[i])
    return WinU
#===========================================
#right-hand-side of DDE dummy
#===========================================
def f(args):
    res = 0.
    for i in range(len(args)):
        res += sin(args[i])
    return res
#===========================================
#preprocessing and activation dummies
#===========================================
def f_in(x: float) -> float:
    return tanh(x)
#==========================================
#preprocessed input
#====================
J1 = J(Win, inp)
#===================
def f_out(x:float) -> bool:
    if x > 20:
        return True
    return False
#===========================================
#Solving DDE <-> Neural Network processing
#===========================================
def sol(iv=0., J1=J1, Wout=Wout):
    output = [None] * outsize
    x = [None] * (L*N)
    # input layer
    #=============================================
    x[0] = iv + theta*f([iv, J1[0]])
    for n in range(1, N):
        x[n] = x[n-1] + theta*f([x[n-1], J1[n]])
    #=============================================
    # hidden layers
    #=============================================
    for l in range(1, L):
        for n in range(N):
            t = l*N + n
            delayed = [0.] * len(delays)
            for k in range(len(delays)):
                if M[t][k] != 0:
                    delayed[k] = M[t][k] * x[t-delays[k]]
            x[t] = x[t-1] + theta*f([x[t-1], z[t]] + delayed)
    #==============================================
    #output layer
    #==============================================
    nn_out = x[(L-1)*N:] + [1.]
    Wout = matr_vec_mult(Wout, nn_out)
    for i in range(outsize):
        output[i] = f_out(Wout[i])
    return output
#===================================================

if __name__=="__main__":
    print(sol())
#print(matr_vec_mult(W_in, inp))