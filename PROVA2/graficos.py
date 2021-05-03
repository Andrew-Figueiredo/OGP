import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from numpy import *

# grafica f em [a,b]x[c,d] em nxn pontos
# igualemente espaçados
def graficoFunEscalar3d(f,a,b,c,d,n):
    x = linspace(a,b,num=n)
    y = linspace(c,d,num=n)
    X = zeros(n*n)
    Y = zeros(n*n)
    for i in range(n):
        X[i*n:(i+1)*n] = x
        Y[i*n:(i+1)*n] = y[i]*ones(n)
    Z=f(X,Y)
    ax = plt.axes(projection='3d')
    ax.plot3D(X,Y,Z,'r.')
    plt.show()
    
# grafica f em [a,b] em n pontos
# igualemente espaçados
def graficoFunEscalar2d(f,a,b,n):
    x=linspace(a,b,n)
    plt.plot(x,f(x))
    plt.grid()
    plt.show()
    
# grafica dataset com X mxn e Y mx1
# se n=1, o gráfico é 2d
# se n=2, o gráfico é 3d
def graficoDataset(X,Y):
    n=X.ndim
    if n==1:
        plt.plot(X,Y,'r.')
    else:
        ax = plt.axes(projection='3d')
        ax.plot3D(X[:,0],X[:,1],Y[:,0],'r.')
