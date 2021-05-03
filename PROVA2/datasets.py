from numpy import *

def carrega(nomeArquivo):
    arquivo = open(nomeArquivo, 'r')
    linhas = arquivo.readlines() 
    nd=len(linhas) # número de linhas
    na=len(linhas[0].strip().split()) # número de colunas
    na=na-1 # número de atributos
    print('\nDataset com m=',nd,'exemplos e n=',na,'atributos.\n')
    X=zeros((nd,na))
    Y=zeros((nd,1))
    i=0
    for linha in linhas:
        linha = linha.strip().split() # retira '\n' e separa entradas
        linha = [float(i) for i in linha]
        X[i,0:na] = linha[0:na]
        Y[i,0] = linha[na]
        i += 1
    return X, Y

