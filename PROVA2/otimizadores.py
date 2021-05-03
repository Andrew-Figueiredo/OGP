from numpy import *
from numpy.linalg import norm, solve, matrix_rank

def armijo(f,df,x,d):
    eta = 0.8
    nu = 0.5
    t = 1.0
    xmtd = x + t*d
    while f(xmtd) > f(x) + d @ df(x)*eta*t:
        t *= nu
        xmtd = x + t*d
    return t

# Método de STEEPEST-DESCENT com buscalinear
# para minimizar f(x) sem restrições
# f  : R^n -> R
def steepestDescent(x0,f,gradf,buscaLinear=armijo,epsilon=1e-5):
    x = x0.copy()
    k = 0
    while True:
        d = -gradf(x)
        nd2 = d@d
        if nd2<epsilon:
            break
        t = buscaLinear(f,gradf,x,d) # busca linear
        print('k=',k,'|d|=',norm(d),'t=',t) # se f aumentar, tem que terminar
        x += t*d
        k += 1
    return x

# Método de GAUSS-NEWTON
# para minimizar |f(x)|^2 sem restrições
# f  : R^n -> R^m
# Jf : R^n -> R^(mxn)
def gaussNewton(x0,f,Jf,TOL1 = 1e-5,TOL2 = 1e-6,MAXITER = 100):
    n = len(x0)
    xs = zeros((MAXITER,n))
    x = x0.copy()
    k = 0
    xs[k,:] = x0
    while 1:
        b = -f(x)
        nfx = norm(b)
        if nfx>1e100:
            print('gaussNewton: |f(x)| muito grande')
            break
        if nfx<TOL1:
            print('Termina por |f(x)|<',TOL1)
            break
        A = Jf(x)
        if matrix_rank(A)<n:
            print('gaussNewton: As colunas da matriz jacobiana de f não são l.i.')
            break
        b = A.T@b
        A = A.T@A
        d = solve(A, b)
        if norm(d)<TOL2:
            print('Termina por |d|<',TOL2)
            break
        x = x + 1.0*d
        k += 1
        xs[k,:] = x
        print('k=',k,'|f(x)|=',nfx,'|d|=',norm(d))
        if k>MAXITER:
            print('Termina por MAXITER')
            break
    xs = xs[0:k+1]
    return x,xs

# Método de LEVENBERG-MARQUARDT
# para minimizar |f(x)|^2 sem restrições
# f  : R^n -> R^m
# Jf : R^n -> R^(mxn)
def levenbergMarquardt(x0,f,Jf,TOL1 = 1e-5,TOL2 = 1e-6,MAXITER = 100):
    n = len(x0)
    xs = zeros((MAXITER,n))
    x = x0.copy()
    lamb = 1
    k = 0
    xs[k,:] = x0
    J = Jf(x)
    fx = f(x)
    while 1:
        b = concatenate((J@x-fx,sqrt(lamb)*x))
        nfx = norm(fx)
        if nfx<TOL1:
            print('Termina por |f(x)|<',TOL1)
            break
        A = vstack((J,sqrt(lamb)*eye(n)))
        b = A.T@b
        A = A.T@A
        xnovo = solve(A, b)
        d = xnovo - x
        if norm(d)<TOL2:
            print('Termina por |d|<',TOL2)
            break
        #xnovo = x + 1.0*d
        if norm(f(xnovo))<nfx:
            x = xnovo
            J = Jf(x)
            fx = f(x)
            lamb *= 0.8
        else:
            lamb *= 10.0
        k += 1
        xs[k,:] = x
        print('k=',k,'|f(x)|=',nfx,'|d|=',norm(d))
        if k>MAXITER:
            print('Termina por MAXITER')
            break
    xs = xs[0:k+1]
    return x,xs

# Método de LEVENBERG-MARQUARDT
# para minimizar |f(x)|^2 sem restrições
# f  : R^n -> R^m
# Jf : R^n -> R^(mxn)
def levenbergMarquardt2(x0,f,Jf,TOL1 = 1e-5,TOL2 = 1e-6,MAXITER = 100):
    n = len(x0)
    xs = zeros((MAXITER,n))
    x = x0.copy()
    lamb = 1
    k = 0
    xs[k,:] = x0
    J = Jf(x)
    fx = f(x)
    while 1:
        b = concatenate((-fx,zeros(n)))
        nfx = norm(fx)
        if nfx<TOL1:
            print('Termina por |f(x)|<',TOL1)
            break
        A = vstack((J,sqrt(lamb)*eye(n)))
        b = A.T@b
        A = A.T@A
        d = solve(A, b)
        if norm(d)<TOL2:
            print('Termina por |d|<',TOL2)
            break
        xnovo = x + 1.0*d
        if norm(f(xnovo))<nfx:
            x = xnovo
            J = Jf(x)
            fx = f(x)
            lamb *= 0.8
        else:
            lamb *= 10.0
        k += 1
        xs[k,:] = x
        print('k=',k,'|f(x)|=',nfx,'|d|=',norm(d))
        if k>MAXITER:
            print('Termina por MAXITER')
            break
    xs = xs[0:k+1]
    return x,xs
