# Exercicio 02

from numpy import zeros

def loadingDataSet(fileName):
    file = open(fileName, 'r')
    lines = file.readlines()
    m = len(lines)
    n = len(lines[0].strip().split()) - 1

    print('\nDataset com m=',m,'exemplos e n=',n,'atributos.\n')
    X = zeros((m,n))
    Y = zeros((m,1))

    i = 0

    for line in lines:
        line = line.strip().split()
        line = [float(i) for i in line]
        X[i, 0:n] = line[0:n]
        Y[i,0] = line[n]
        i += 1
    return X, Y


X, Y = loadingDataSet('./dataset01.txt')
m,n = X.shape


if n==1:
    import matplotlib.pyplot as plt
    plt.plot(X,Y,'r.')
    plt.grid()
    plt.show()
else:
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    ax = plt.axes(projection='3d')
    ax.plot3D(X[:,0],X[:,1],Y[:,0],'r.')
    plt.show()