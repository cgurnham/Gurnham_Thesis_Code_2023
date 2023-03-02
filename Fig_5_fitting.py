import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import scipy as sp

import lmfit


Tc = 90 #in kelvin, superconducting transition temperature

THEVAT = [77, 77, 77, 77, 77, 77, 77]
THEVAB = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
THEVAJ = [4.274, 4.70365, 5.28067, 5.96541, 7.00157, 8.80211, 12.64651]


FujiT = [77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2,]
FujiB = [0.25, 0.5, 0.75, 1, 1.5, 1.75, 2, 2.25, 2.75, 3, 2, 3, 4, 5, 6, 8, 10, 11, 14, 16, 18]	
FujiJ = [11.33, 8.034214206, 6.301300336, 5.323042506, 4.037332215, 3.47832774, 3.003173937, 2.611870805, 2.164667226, 1.913115213, 133.5198372, 107.5788403, 91.30213632, 80.11190234, 72.22787386, 60.52899288, 52.3906409, 46.66836216, 41.83621567, 38.27568667, 34.96948118]


SunamT = [77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2, 4.2]
SunamB = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18]
SunamJ = [12.9483035, 7.209190902, 5.0849739, 4.041498881, 3.109824758, 2.550820283, 2.178150634, 1.880014914, 1.54461223, 1.39554437, 1.24647651, 1.02287472, 157.5110207, 105.6290268, 70.70193286, 56.12071889, 46.79552391, 40.35266192, 36.11393693, 30.34927094, 26.78874195, 24.07595795, 21.70227196, 20.34587996, 19.66768396]


    
def JcTfit(Temp, B, d_thick, ws, c1, Tcn, M0, beta, w, Bc20):
    c0 = np.exp((c1-0.759)/0.181)
    #Bc2 = Bc20*(1-(Temp/Tc)**w)   #Calculate Bc2 at a given temperature - turned off, see next line
    Bc2 = -w*np.heaviside(Temp - 20, 0) + 115 #This sets the Bc2 to either 8 T (T=77 K) or 115 T (T=4.2 K). I don't have enough data points to fit w so I have used this to fix Bc2 at the two temperatures
    xi = 18.09/np.sqrt(Bc2)
    d = d_thick/xi
    M_nu = M0/(Bc2/(Bc20*(1-(Temp/Tc)**2))) #calculate m / upsilon ratio at given temp
    alpha = (Tcn/Tc)*(1-(Temp/Tcn))/(1-(Temp/Tc)**2) #calculate alpha from given T
    J0 = 10901*np.sqrt(Bc20/Bc2)*(1-(Temp/Tc)**2)**2
    B0 = (1/(ws**2))*(c0**(1/c1))
    
    def s(x):
        return beta*(1-x)/(alpha-(x/M_nu))
  
    def Xi_n(y): #normalised to xi_s
        
        return 1/np.sqrt(np.abs(y-(alpha*M_nu)))
  
    def nu(z):
        return M_nu*(Xi_n(z))*np.sqrt(1-z)

    def fd_2(a):
        return (((nu(a))**2)+1-np.sqrt(((nu(a))**2)*(2-s(a))+1))/(((nu(a))**2)+s(a))


    def Jd(b):
        return 4*J0*((1-b)**1.5)*(1-np.sqrt(1-(s(b))*fd_2(b)))*np.exp(-d/(Xi_n(b)))/(s(b)*nu(b))
            
    
    def Jc(B):
        return Jd(B/Bc2)*c0*(np.tanh(B/B0)/(B*(ws**2)))**c1
    
    
    return Jc(B)
    
Brange = np.logspace(-3, 2, 100)

Jcfit1 = np.zeros(len(Brange))
Jcfit2 = np.zeros(len(Brange))


model = lmfit.Model(JcTfit,independent_vars=['B','Temp'],param_names=['ws','c1','Tcn', 'M0', 'beta', 'd_thick', 'w', 'Bc20'])

ws_lowerbound=2.2
wsfree='yes'
wsval=4.5
model.set_param_hint('ws', value= wsval, vary=True, min=ws_lowerbound)

c1_lowerbound=0.56
c1free='yes'
c1val=0.62
model.set_param_hint('c1', value= c1val, vary=True, min=c1_lowerbound)

Tcn_upperbound = 4
Tcnfree='yes'
Tcnval=1.27
model.set_param_hint('Tcn', value= Tcnval, vary=False, max = Tcn_upperbound)


M0free='no'
M0val=4.56
model.set_param_hint('M0', value= M0val, vary=False)

beta_lowerbound = 0
betafree='yes'
betaval=1E-9
model.set_param_hint('beta', value= betaval, vary=False, min = beta_lowerbound)

d_thick_lowerbound=0
d_thickfree='yes'
d_thickval=3
model.set_param_hint('d_thick', value= d_thickval, vary=False, min=d_thick_lowerbound)

w_lowerbound=0.1
w_upperbound = 110
wfree='yes'
wval=107
model.set_param_hint('w', value= wval, vary=False, min=w_lowerbound, max = w_upperbound)

Bc20_lowerbound = 50
Bc20_upperbound = 250
bc2free = 'yes'
bc20val = 115
model.set_param_hint('Bc20', value= bc20val, vary=False) #, min=Bc20_lowerbound, max = Bc20_upperbound)


A_Model_params = model.make_params()
Jc_points_for_fit= THEVAJ#SunamJ #FujiJ
B_points_for_fit=THEVAB #SunamB #FujiB
T_points_for_fit=  THEVAT#SunamT #FujiT
fit_weights = np.ones(len(Jc_points_for_fit))# + np.reciprocal(Jc_points_for_fit)#+ 
fit_results=model.fit(data=Jc_points_for_fit,params=A_Model_params,B=B_points_for_fit,Temp=T_points_for_fit,weights=fit_weights,nan_policy='propagate',scale_covar='False')

fitvals=np.array(())
for key in fit_results.params:
    fitvals=np.append(fitvals,fit_results.params[key].value)

ws_fitval = fitvals[0]
c1_fitval = fitvals[1]
Tcn_fitval = fitvals[2]
M0_fitval = fitvals[3]
beta_fitval = fitvals[4]
d_thick_fitval = fitvals[5]
w_fitval = fitvals[6]
Bc20_fitval = fitvals[7]



for i in range(len(Brange)):
    Jcfit1[i] = JcTfit(77, Brange[i], d_thick_fitval, ws_fitval, c1_fitval, Tcn_fitval, M0_fitval, beta_fitval, w_fitval, Bc20_fitval) #plot the fit at T = 77 K
    
    Jcfit2[i] = JcTfit(4.2, Brange[i], d_thick_fitval, ws_fitval, c1_fitval, Tcn_fitval, M0_fitval, beta_fitval, w_fitval, Bc20_fitval) #plot the fit at T = 4.2 K
    

print('Bc2 at 77 K = ', Bc20_fitval*(1-(77/90)**w_fitval), 'Bc2 at 4.2 K = ', Bc20_fitval*(1-(4.2/90)**w_fitval))
print(fitvals)
plt.scatter(THEVAB, THEVAJ)
plt.plot(Brange, Jcfit1)
plt.plot(Brange, Jcfit2)
plt.xscale("log")
plt.yscale("log")
plt.show()