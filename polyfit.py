
import pandas as pd 			#importa pandas como pd
import matplotlib.pyplot as plt		#importa pyplot como plt
import numpy as np 			#importa numpy como np
import sympy as S 			#importa sympy como S

file = 'exam_B_dataset.csv' 		#Se elige el archivo de datos y se asigna a file y se asignan a sus respectivos campos X o Y
data = np.loadtxt(file, delimiter = ',', skiprows = 0, usecols=[0,1])
X = np.loadtxt(file, delimiter= ',' ,skiprows = 0, usecols= [0])
Y = np.loadtxt(file, delimiter= ',' ,skiprows = 0, usecols= [1])



def polyfit2(x,y,n):			#declara la función polyfit2 con sus argumentos.

    def inv(A):    			#funcion que realiza la inversa de A
        return np.linalg.inv(A)
    def trans(A): 			#funcion que realiza la transpuesta de A
        return A.getT()
    def oneMat(xl,n):  			#realiza una matriz de unos por numpy
        return np.ones((xl,n),dtype=int)
    def prod(A,B): 			#funcion que realiza el producto de A y B
        return np.dot(A,B)

   				 
    xlen = len(x)			#toma la longitud del vector x' y lo guarda en xlen
    ylen = len(y)			#toma la longitud del vector y' y lo guarda en ylen
    one = np.ones((xlen,n+1),dtype=int)	#crea la matriz de unos tomando en cuenta la longitud del vector x (xlen)
    c1=one[:,[1]]			#los valores de la primera columna son los 'unos de la matriz one
    xT=np.matrix(x)			#crea la matriz de x
    yT=np.matrix(y)			#crea la matriz de y
    c2=xT.getT()			#pone en la segunda columna los valores de x(xT)
    c3=np.power(c2,2)			#pone en la tercera columa el valor de la segunda columna, al cuadrado.
    A=np.hstack([c1,c2,c3])		#junta las columnas para la creación de la matriz

    return prod(prod(inv(prod(trans(A),A)),trans(A)),trans(yT))   #realiza la ecuación completa para el cálculo del ajuste
print(polyfit2(X,Y,2)) 

x = S.symbols('x') 
y = S.symbols('y')
 
#una vez que se obtienen los valores de ajuste, se realizan los cálculos para representar los puntos de ajuste en la gráfica

y = -3.33 + 0.53 * x -1.94 * pow(x,2) +0.523 * pow(x,3) +0.496 * pow(x,2) #
f = S.lambdify(x,y,'math')
yen = f(X)
print(yen)
yout= yen.astype(list)

plt.scatter(X,Y, color = 'orange')
plt.scatter(X,yout , color='purple')
plt.show()





def polyfit3(x,y,n): 

    def inv(A): 
        return np.linalg.inv(A)
    def trans(A):
        return A.getT()
    def oneMat(xl,n): 
        return np.ones((xl,n),dtype=int)
    def prod(A,B): 
        return np.dot(A,B)

    xlen = len(x)
    ylen = len(y)
    one = np.ones((xlen,n+1),dtype=int)
    c1=one[:,[1]]
    xT=np.matrix(x)
    yT=np.matrix(y)
    c2=xT.getT()
    c3=np.power(c2,2)
    c4 = np.power(c2,3) 

    A=np.hstack([c1,c2,c3,c4])


    return prod(prod(inv(prod(trans(A),A)),trans(A)),trans(yT))
print(polyfit3(X,Y,3)) 

x = S.symbols('x')
y = S.symbols('y')


y = -3.33 + 0.53 * x -1.94 * pow(x,2) +0.523 * pow(x,3) +0.496 * pow(x,3)  

f = S.lambdify(x,y,'math')
yen = f(X)
print(yen)
yout= yen.astype(list)

plt.scatter(X,Y, color = 'orange')
plt.scatter(X,yout , color='purple')
plt.show()





def polyfit4(x,y,n): 

    def inv(A): 
        return np.linalg.inv(A)
    def trans(A):
        return A.getT()
    def oneMat(xl,n): 
        return np.ones((xl,n),dtype=int)
    def prod(A,B): 
        return np.dot(A,B)

    xlen = len(x)
    ylen = len(y)
    one = np.ones((xlen,n+1),dtype=int)
    c1=one[:,[1]]
    xT=np.matrix(x)
    yT=np.matrix(y)
    c2=xT.getT()
    c3=np.power(c2,2)
    c4 = np.power(c2,3)
    c5= np.power(c2,4) 

    A=np.hstack([c1,c2,c3,c4,c5])

    return prod(prod(inv(prod(trans(A),A)),trans(A)),trans(yT))
print(polyfit4(X,Y,4))

x = S.symbols('x')
y = S.symbols('y')


y = -3.33 + 0.53 * x -1.94 * pow(x,2) +0.523 * pow(x,3) +0.496 * pow(x,4) 

f = S.lambdify(x,y,'math')
yen = f(X)
print(yen)
yout= yen.astype(list)


plt.scatter(X,Y, color = 'orange')
plt.scatter(X,yout , color='purple')
plt.show()
