import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import *

def kn(n):
    return (n+0.5)*2*np.pi
    
def summation(y):  #Evaluate the phase difference in the junction
    n = 0
    output = 1
    total = 0
    while output > 1E-17 or n < 500:
        output  = ((-1)**n/(kn(n)**3))*np.tanh(kn(n)*Aspect/4)*np.sin(kn(n)*y)
        total += output
        n+=1
    return total
    

def FindMaxima(numbers):
  maxima = []
  length = len(numbers)
      
  for i in range(1, length-1):     
    if numbers[i] > numbers[i-1] and numbers[i] > numbers[i+1]:
        maxima.append(i)       
  return maxima

    
def fitting(x, c0, c1):
    return (c0*(np.tanh(x/((c0/J0)**(1/c1)))/x)**(c1))    

J0 = 1
Arange = np.logspace(-1, 1, num = 3) #L/W - range of aspect ratios you want to output c0 & c1 at
B = 1.0 #in units of phi_0/W
Bmin = 0.01 #range of field points that are fitted over
Bmax = 100
Blength = 5000
Brange = np.linspace(Bmin, Bmax, num = Blength)
Jc = np.empty(len(Brange))
yrange = np.linspace(0, 0.5, num = 3000) #number of points across the junction that are calculated at - this needs to be larger if you increase the magnitude of the field
phasediff = np.empty(len(yrange))
y = 0

    
c0values = []
c1values = []
c1errors = []


for A in Arange:
    Aspect = A*2 #Difference in definition in Clem - need this factor of two to reproduce the same equations
    for i in range(len(yrange)):
        y = yrange[i]
        phasediff[i] = summation(y)

    for i in range(Blength):
        B = Brange[i]
        Jc[i] = abs(np.sum((np.cos((16*np.pi*B)*phasediff))))/len(yrange)
      
     

    
    Maxcoords = FindMaxima(Jc)
    Maxfields = np.empty(len(Maxcoords))
    Maxcurrents = np.empty(len(Maxcoords))
    for i in range(0, len(Maxcoords)):
        Maxfields[i] = Bmin + Maxcoords[i]*(Bmax-Bmin)/Blength
        Maxcurrents[i] = Jc[Maxcoords[i]]
    

    params, params_covariance = sp.optimize.curve_fit(fitting, Maxfields, Maxcurrents, bounds=(0.01, 10)) #ensure c0 & c1 are positive
    c0values.append(params[0])
    c1values.append(params[1])
    c1errors.append(params_covariance[1])



print("c1 = ", c1values)
print("c0 = ", c0values)
print("c1 errors = ", c1errors)

plt.plot(Arange, c0values)
plt.plot(Arange, c1values)
plt.xscale("log")
plt.yscale("log")

plt.show()
    

