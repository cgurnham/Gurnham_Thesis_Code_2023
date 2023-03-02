import scipy as sp
import numpy as np
import matplotlib.pyplot as plt


def kn(n):
    return (n+0.5)*2*np.pi
    
def summation(y):
    n = 0
    output = 1
    total = 0
    while output > 1E-17 or n < 500:
        output  = ((-1)**n/(kn(n)**3))*np.tanh(kn(n)*A/4)*np.sin(kn(n)*y)
        total += output
        n+=1
    return total
    


def fitting(x, c0, c1): #plots a fit line for a given value of c0 & c1
    return c0*(np.tanh(x/(c0**(1/c1)))/x)**(c1)


B = 0.01 #in units of phi_0/W^2
Bmin = 0.01 #range of field points that are fitted over
Bmax = 100
Blength = 5000

Brange = np.linspace(Bmin, Bmax, num = Blength)
Jc = np.empty(len(Brange))
yrange = np.linspace(0, 0.5, num = 3000) #number of points across the junction that are calculated at - this needs to be larger if you increase the magnitude of the field
phasediff = np.empty(len(yrange))
y = 0

c0 = 0.574
c1 = 0.576

A = 2 #twice the aspect ratio (A=2 is L = W), discrepancy from Clem's definition
for i in range(len(yrange)):
    y = yrange[i]
    phasediff[i] = summation(y)

for i in range(Blength):
    B = Brange[i]
    Jc[i] = abs(np.sum((np.cos((16*np.pi*B)*phasediff))))/len(yrange)
        
     

plt.xscale("log")
plt.yscale("log")


plt.plot(Brange, Jc)
plt.plot(Brange, fitting(Brange, c0, c1))

plt.show()
    

