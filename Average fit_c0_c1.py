import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpmath import *
from scipy.optimize import *


def term(n):
    return tanh((2*n + 1)*0.25*(np.pi)*Aspect)/((2*n +1)**3)
    

def a0(): #calculate the node spacing
    return (np.pi**3)/(16*nsum(lambda n: term(n), [0, inf]))

def kn(n):
    return (n+0.5)*2*np.pi
    
def summation(y): #Evaluate the phase difference in the junction
    n = 0
    output = 1
    total = 0
    while output > 1E-17 or n < 500:
        output  = ((-1)**n/(kn(n)**3))*np.tanh(kn(n)*Aspect/4)*np.sin(kn(n)*y)
        total += output
        n+=1
    return total
    


def fitting(x, c0, c1): #fitting over log-space to ensure good fit at high-fields
    return np.log(c0*(np.tanh(x/((c0/J0)**(1/c1)))/x)**(c1))


Arange = np.logspace(0, 1, num = 3) #L/W - range of aspect ratios you want to output c0 & c1 at

B = 0.01 #in units of phi_0/W^2
Bmin = 0.01  #range of field points that are fitted over
Bmax = 100
Blength = 5000 

Brange = np.linspace(Bmin, Bmax, num = Blength)
Jc = np.empty(len(Brange))

yrange = np.linspace(0, 0.5, num = 3000)   #number of points across the junction that are calculated at - this needs to be larger if you increase the magnitude of the field
phasediff = np.empty(len(yrange))
y = 0

c0values = []
c1values = []
c0errors = []
c1errors = []


for A in Arange:
    Aspect = A*2 #Difference in definition in Clem - need this factor of two to reproduce the same equations
    a0val = a0()
    pointsrange = Blength*a0val/(2*(Bmax-Bmin)) #half of the node-spacing
    
    ir = int(pointsrange)
    Jcave = np.empty(len(Brange)-ir)
    Bave = np.empty(len(Brange)-ir)
   
    for i in range(len(yrange)):
        y = yrange[i]
        phasediff[i] = summation(y)

    for i in range(Blength):
        B = Brange[i]
        Jc[i] = abs(np.sum((np.cos((16*np.pi*B)*phasediff))))/len(yrange)
     
    
    for j in range(Blength-ir): #find the Jc averaged over one node-spacing
        Jcavesum=0
        if j > ir - 1:
            for k in range(j-ir,j+ir):
                Jcavesum +=Jc[k]
        else: #integrate over negative values as well
            Jcavesum = 0
            for k in range(0,j+ir):
                Jcavesum +=Jc[k]
            for k in range(0,ir-j):
                Jcavesum +=Jc[k]
            
        Jcave[j] = Jcavesum/(2*ir)
        J0 = 1 #constrain the fit to have the correct zero-field limit
        Bave[j] = Brange[j]
    params, params_covariance = sp.optimize.curve_fit(fitting, Bave, np.log(Jcave), bounds=(0.01, 100)) #ensure c0 & c1 are positive
    c0values.append(params[0])
    c1values.append(params[1])
    c0errors.append(params_covariance[0])
    c1errors.append(params_covariance[1])
    



print("c1 = ", c1values)
print("c0 = ", c0values)
print("c1 errors = ", c1errors)
print("c0 errors = ", c0errors)

plt.plot(Arange, c0values)
plt.plot(Arange, c1values)
plt.xscale("log")
plt.yscale("log")

plt.show()
    

