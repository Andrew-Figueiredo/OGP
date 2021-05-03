from datasets import carrega
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt
from graficos import graficoDataset, graficoFunEscalar2d
from numpy import squeeze, ones, array
from otimizadores import steepestDescent, gaussNewton, levenbergMarquardt

X, Y = carrega('dataset08.txt') # faz a leitura do arquivo e define o dataset
m = X.shape[0]
X = squeeze(X) # vou usar array 1d
Y = squeeze(Y)

def fpred(x,p):
    a = p[0]
    b = p[1]
    c = p[2]
    d = p[3]
    return a*x**3+b*x**2+c*x+d

def gradFpred(x,p):
    A = array([x**3, x**2, x, ones(m)])
    return A.T

# empregadas no gaussNewton e levenbergMarquardt
def f(p):
    return fpred(X,p)-Y

def Jf(p):
    return gradFpred(X,p)


# empregadas no steepestDescent
def fobj(p):
    v = f(p)
    return v.T@v

def gradfobj(p):
    return 2*f(p) @ Jf(p)



p0 = array([4.01,-1.05,3.1,2.0]) # pode divergir

p,ps = levenbergMarquardt(p0,f,Jf)
print(p)

graficoDataset(X,Y)
g = lambda x: fpred(x,p)
graficoFunEscalar2d(g,min(X),max(X),m)