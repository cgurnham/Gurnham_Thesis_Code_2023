import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

T = 15 #operating temperature
#Bc2 = 8000 #in mTesla, at operating temp
t = 2 #thickness of tape in microns

beta = 1E-9
Tcn = 0.990926
Tc = 92
d_thick = 3.309121 #junction thickness in nm
sw = 1.34531
Bc20 = 120000 #in mTesla
Bc2 = Bc20*((1-T/Tc)**sw)


xi = 18.09/np.sqrt(Bc2/1000) #in nm
d = d_thick/xi #junction thickness in units of xi

c1 = 0.56 #field exponent for unity aspect ratio

c0 = np.exp((c1-0.759)/0.181)
ws = 0.901 #in units of sqrt(phi_0) - 2.2 is 100 nm
B0 = 1000*(1/(ws**2))*(c0**(1/c1)) #in mT

M_nu = 2.471/(Bc2/(Bc20*(1-(T/Tc)**2))) #calculate m / upsilon ratio at given temp

alpha = (Tcn/Tc)*(1-(T/Tcn))/(1-(T/Tc)**2) #calculate alpha from given T
J0 = 10901*np.sqrt(Bc20/Bc2)*(1-(T/Tc)**2)**2 #calculate J0 from given T

def s(x):
    return beta*(1-x)/(alpha-(x/M_nu))
  
def Xi_n(y):
    return 1/np.sqrt(y-alpha*M_nu)
  
def nu(z):
    return M_nu*Xi_n(z)*np.sqrt(1-z)

def fd_2(a):
    return (((nu(a))**2)+1-np.sqrt(((nu(a))**2)*(2-s(a))+1))/(((nu(a))**2)+s(a))


def Jd(m): #takes reduced field B/Bc2
    b = abs(m)
    return 4*J0*((1-b)**1.5)*(1-np.sqrt(1-(s(b))*fd_2(b)))*np.exp(-d/(Xi_n(b)))/(s(b)*nu(b))
    
    
def Jc(B): #takes Bapp in mT rather than b*
    return (Jd(B/Bc2)*c0*(np.tanh(abs(B)/B0)/((B/1000)*(ws**2)))**c1)



No_elements = 51 #number of elements across width of tape

#Bext = 100 
Current = 0

def func(B):
    Bfunc = Bext*np.ones(No_elements)
    for m in range(No_elements):
        for n in range(No_elements):
            if abs(m-n) > 0.1:
                Bfunc[m] += (0.2*t*Jc(abs(B[n]))/(n-m))
            else:
            
                #if No_elements-2.1 > n and n > 1.1: #1st order derivs, adjacent only
                #    Bfunc[m] -= 0.2*(1/2 - (np.log(3)-1))*t*Jc(abs(B[n+1]))
                #    Bfunc[m] += 0.2*(1/2 - (np.log(3)-1))*t*Jc(abs(B[n-1]))
                #    Bfunc[m] -= 0.2*((np.log(3)-1)/2)*t*Jc(abs(B[n+2]))
                #    Bfunc[m] += 0.2*((np.log(3)-1)/2)*t*Jc(abs(B[n-2]))
                #if No_elements-3.1 > n and n > 2.1: #1st order derivs, calculated from 2 points away
                #    Bfunc[m] += 0.2*(-2/3+(np.log(3)-1)*13/12)*t*Jc(abs(B[n+1]))
                #    Bfunc[m] -= 0.2*(-2/3+(np.log(3)-1)*13/12)*t*Jc(abs(B[n-1]))
                #    Bfunc[m] += 0.2*(-1/12-(np.log(3)-1)*2/3)*t*Jc(abs(B[n+2]))
                #    Bfunc[m] -= 0.2*(-1/12-(np.log(3)-1)*2/3)*t*Jc(abs(B[n-2]))
                #    Bfunc[m] += 0.2*((np.log(3)-1)/12)*t*Jc(abs(B[n+3]))
                #    Bfunc[m] -= 0.2*((np.log(3)-1)/12)*t*Jc(abs(B[n-3]))
                #if No_elements-1.1 > n and n > 0.1: #2nd order derivs, calculated at adjacent points only
                #    Bfunc[m] -= 0.1*t*Jc(abs(B[n+1]))
                #    Bfunc[m] += 0.1*t*Jc(abs(B[n-1]))
                    
                if No_elements-2.1 > n and n > 1.1: #2nd order derivs, calculated from 2 points away
                    Bfunc[m] -= 0.2*(2/3 + 4*(np.log(3)-1)/3)*t*Jc(abs(B[n+1]))
                    Bfunc[m] += 0.2*(2/3 + 4*(np.log(3)-1)/3)*t*Jc(abs(B[n-1]))    
                    Bfunc[m] += 0.2*(1/12 + 2*(np.log(3)-1)/3)*t*Jc(abs(B[n+2]))
                    Bfunc[m] -= 0.2*(1/12 + 2*(np.log(3)-1)/3)*t*Jc(abs(B[n-2]))
                
                
                Bfunc[m] -= B[m]
    return(Bfunc)      

B0 = 1000*(1/(ws**2))*(c0**(1/c1))
Bapp = 2000 #Applied field in mT
Bext = Bapp
  
#d = 0.13181*np.log(9178.8/((ws*45.45)**1.12))   #empirical rescaling of thickness to allow for changes of junction width while conserving in-field Jc. Turn this off if you want to use the fit parameters (d calculated at line 17)
   #I have used this to generate figures 3 & 4. For data with actual fit values (fig 5) - make sure to comment this line out and enter the appropriate fit values above.
Barray = fsolve(func,-1*np.linspace(-50+Bapp,49.1+Bapp,No_elements)) #initialise with linear field distribution
Jarray = np.zeros(len(Barray))
posarray = np.linspace(-1,1,No_elements)
    
for m in range(len(Barray)):
    Jarray[m] = Jc(abs(Barray[m]))
    
    
Current = np.average(Jarray) #Calculates the average current across the tape
plt.plot(posarray, Barray)
plt.plot(posarray, Jarray)
plt.show()
print(Current) 
print("B = ", Bapp, "mT")
