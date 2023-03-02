from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from scipy.odr import ODR, Model, Data, RealData
from scipy import stats

os.chdir("C:\\Users\\Mark\\OneDrive - Durham University\\Documents\\Charles\\Data_Nov21")
raw_array=np.genfromtxt("06_02T_90deg", delimiter='\t', usecols = (0,1,4), skip_header=20, skip_footer = 2)

#temps_array=raw_array[:,2:5]
current_voltage_array=raw_array[:,0:2]
shunt_diff_grad, shunt_diff_int=0.00864, 0.003


baseline_corr_order=1
springboard_shunt_resistance =1000 
vtap_length=9#in mm
vtap_length_error = 0.3 #in mm
trace_baseline_beginning=0.0 #between 0 and 1
trace_baseline_end=0.7
trace_transition_start = 0.7
g=49.39 #Gain of Amplifier (*10**3)
g_err=0.04 #Error on Amplifier Gain
V_Inst_err=0.5 #Micro-volts, error on Keithley 2100 multimeter in relevant measuring range
I_Inst_err=0.05 #Amps
R=50.42 #Micro-ohms, resistance of shunt to calc I
R_err=0.01 #Micro-ohms, error on shunt resistance

def linear(beta,x):
    return beta[0] + beta[1]*x
    
def quadratic(beta,x):
    return beta[0] + beta[1]*x + beta[2]*x**2

#def calculate_voltage_datapoint_errors(V_rec):
#    '''V_rec must be in microvolts'''
#    
#    
#    err_sqrd=np.zeros(np.size(V_rec))
#    for i in range(np.size(V_rec)):
#        err_sqrd[i]=V_rec[i]**2*((V_Inst_err/(V_rec[i]*g*10**3))**2+(g_err/g)**2)
#    return err_sqrd**0.5 #In units of microvolts

#def calculate_current_datapoint_errors(I_rec):
#    '''I_rec must be in amps'''
#    
#   
#   
#    err_sqrd=np.zeros(np.size(I_rec))
#    for i in range(np.size(I_rec)):
#        err_sqrd[i]=I_rec[i]**2*((I_Inst_err/(R*I_rec[i]))**2+(R_err/R)**2)
#    return err_sqrd**0.5

def correct_current_data(currents,gradient,intercept):
    for i in range(len(currents)):
        currents[i]=currents[i]-gradient*currents[i]-intercept
    return currents

def remove_springboard_shunting(currents,voltages,shunt_resistance):
    corrected_currents=np.zeros(np.size(currents))
    for i in range(np.size(corrected_currents)):
        corrected_currents[i]=currents[i]-voltages[i]*10**-6/shunt_resistance
    return corrected_currents

def data_correction(data, corr_order, tap_length, shunt_res,trace_baseline_beginning,trace_baseline_end):
    data = np.array(data)
    current = data[:,0]
    voltage = data[:,1]
    #current=remove_springboard_shunting(current,voltage,shunt_res)
    current_err=I_Inst_err*np.ones(len(current))
    voltage_err=V_Inst_err*np.ones(len(current))
   
    correction=0
    fit_start_idx = int(trace_baseline_beginning*len(current))+10 #To stop erroneous starting data points
    fit_end_idx   = int(trace_baseline_end*len(current))
    
    odrdata=RealData(current[fit_start_idx:fit_end_idx],voltage[fit_start_idx:fit_end_idx],current_err[fit_start_idx:fit_end_idx],voltage_err[fit_start_idx:fit_end_idx])
    model=0
     
    if corr_order == 1:
        model=Model(linear)
        odr=ODR(odrdata,model,[0,0])
        odr.set_job(fit_type=0)
        output=odr.run()
        fit=output.beta
        correction = fit[1]*current + fit[0]
        voltage -= correction
    
    elif corr_order == 2:
        model=Model(quadratic)
        odr=ODR(odrdata,model,[0,0,0])
        odr.set_job(fit_type=0)
        output=odr.run()
        fit=output.beta
        correction = fit[2]*current**2 + fit[1]*current + fit[0]
        voltage -= correction
    
    I_0=(trace_baseline_end-trace_baseline_beginning)*np.max(current)
    
    E_field        = (voltage/tap_length)*1000
    corr_data      = np.zeros([len(current),3])
    corr_data[:,0] = current #in A
    corr_data[:,1] = voltage #in uV
    corr_data[:,2] = E_field #in uV/m
    return corr_data, correction, fit_start_idx, fit_end_idx, output.sd_beta, I_0, fit

def calculate_baseline_subtraction_err_lin(Ic,n,Vc,u_a_B,u_b_B,I_0):
    return (Ic/(n*Vc))*(u_a_B**2+(u_b_B*(Ic-I_0))**2)**0.5

def calculate_baseline_subtraction_err_quad(Ic,n,Vc,u_a_B,u_b_B,u_c_B,I_0):
    return Ic/(n*Vc)*(u_a_B**2+u_b_B**2*(Ic-I_0)**2+u_c_B**2*(Ic-I_0)**4)**0.5

def power_law_fit_err(Ic,u_A,u_n,Vc,ave_ln_V):
    return(Ic*(u_A**2+1))

def analysis(corr_data,baseline_corr_ord,baseline_errors,I_0, baselineparams):
    temp3 = False
    
    for m in range(int(trace_transition_start*len(corr_data[:,0])), len(corr_data[:,0])):    #find the first data point above the 10 uV/m criterion (ie 0.13 microvolts)
        
        if corr_data[m,1] > 0.01*vtap_length and temp3 == False:
            temp3 = True
            tran_start_idx = m
        elif corr_data[m,1] > 0.1*vtap_length and temp3 == True:    #find the last data point below the 100 uV/m criterion (ie 1.3 microvolts)
            tran_end_idx = m-1
            break
    
    transition = corr_data[tran_start_idx:tran_end_idx,:]#isolate the data in the transition region
    for i in range(len(transition)): #remove negative voltages
        if transition[i,1] < 0.0001:
            transition [i,1] = 0.0001

    #print(transition)
    current_errs=I_Inst_err
    voltage_errs=V_Inst_err
    print(transition[:,0])
    log_currents=np.log(transition[:,0])
    log_voltages=np.log(transition[:,1])
    log_currents_errs=I_Inst_err/transition[:,0]
    log_voltages_errs=V_Inst_err/transition[:,1]
    #print(log_currents_errs)
    
    I_0=(trace_baseline_end-trace_baseline_beginning)*np.max(corr_data[:,0])
   
    
    odrdata=RealData(log_voltages,log_currents,log_voltages_errs,log_currents_errs)
    model=Model(linear)
    odr=ODR(odrdata,model,[0,0])  
    output=odr.run()
    tran_fit =output.beta 
    tran_fit_errs=output.sd_beta
    inv_n_value = tran_fit[1]
    n_value=1/inv_n_value
    A=tran_fit[0]
    ave_lnV=np.average(log_voltages)
    
    Ic100   = np.exp(np.log((0.1*vtap_length)**inv_n_value)+A)
    Ic10    = np.exp(np.log((0.01*vtap_length)**inv_n_value)+A)
    
    #Calculating the Errors on the Ic Values

    #Calculate the N-value Error
    u_nval=np.sqrt((tran_fit_errs[1]/(inv_n_value**2))**2+(baseline_errors[0]/(np.log(Ic100)*0.1*vtap_length))**2+((Ic100-I_0)*baseline_errors[1]/(np.log(Ic100)*0.1*vtap_length))**2)
    #print(baselineparams)
    #u_nval=np.sqrt((tran_fit_errs[1]/(inv_n_value**2))**2+(baseline_errors[0]/(inv_n_value*0.1*vtap_length))**2+((Ic100-I_0)*baseline_errors[1]*(1+1/inv_n_value)/0.1*vtap_length)**2+(vtap_length_error/(inv_n_value*vtap_length))**2+(g_err/(inv_n_value*g))**2+(V_Inst_err/inv_n_value*0.1*vtap_length)**2)


   #Baseline Correction Error (linear)
    if baseline_corr_ord==1:               
        u_100_1=calculate_baseline_subtraction_err_lin(Ic100,n_value,0.1*vtap_length,baseline_errors[0],baseline_errors[1],I_0)
        u_10_1=calculate_baseline_subtraction_err_lin(Ic10,n_value,0.01*vtap_length,baseline_errors[0],baseline_errors[1],I_0)
        
    #Baseline Correction Error (Quadratic)
    elif baseline_corr_ord==2:             
        u_100_1=calculate_baseline_subtraction_err_quad(Ic100,n_value,0.1*vtap_length,baseline_errors[0],baseline_errors[1],baseline_errors[2],I_0)
        u_10_1=calculate_baseline_subtraction_err_quad(Ic10,n_value,0.01*vtap_length,baseline_errors[0],baseline_errors[1],baseline_errors[2],I_0)
    
    #Power Law Fitting Error
    
    u_100_2 = Ic100*tran_fit_errs[0]
    u_10_2 = Ic10*tran_fit_errs[0]
    #u_100_2=Ic100*(tran_fit_errs[0]**2+tran_fit_errs[1]**2*(np.log(0.1*vtap_length)-ave_lnV)**2)**0.5                             
    #u_10_2=Ic10*(tran_fit_errs[0]**2+tran_fit_errs[1]**2*(np.log(0.01*vtap_length)-ave_lnV)**2)**0.5  
    
    #Voltage Measurement Error
    u_100_3=(Ic100/n_value)*((vtap_length_error/vtap_length)**2+(g_err/g)**2)**0.5                             
    u_10_3=(Ic10/n_value)*((vtap_length_error/vtap_length)**2+(g_err/g)**2)**0.5  
    
    
    #Current Measurement Error
    u_100_4=(1+baselineparams[1]*Ic100/(n_value*0.1*vtap_length))*(I_Inst_err**2+(R_err*Ic10/R)**2)**0.5                             
    u_10_4=(1+baselineparams[1]*Ic10/(n_value*0.01*vtap_length))*(I_Inst_err**2+(R_err*Ic10/R)**2)**0.5    
    
    #N-value error contribution to Ic error
    u_100_5 = Ic100*np.log(10)*u_nval/(2*n_value**2)
    u_10_5 = Ic10*np.log(10)*u_nval/(2*n_value**2)
    
    
    #Calculate the Total Error
    u_100=(u_100_1**2+u_100_2**2+u_100_3**2+u_100_4**2+u_100_5**2)**0.5
    u_10=(u_10_1**2+u_10_2**2+u_10_3**2+u_10_4**2+u_10_5**2)**0.5
    
    print(u_100_1, u_100_2, u_100_3, u_100_4, u_100_5)
    
    
    return Ic100, Ic10, n_value, transition, u_100, u_10, u_nval

Board_voltage = stats.linregress(raw_array[:,0],raw_array[:,2])



#'''120 A Supply current correction'''
#ps_correct_required=input('Is the power supply the 120 A IPS? (y/n)\n')
#if ps_correct_required =='y':
#    current_voltage_array[:,0]=correct_current_data(current_voltage_array[:,0],shunt_diff_grad,shunt_diff_int)

#baseline_corr_order=float(input('What is the baseline correction order required? (1,2)\n'))

baseline_corrected_data, voltage_correction, baseline_start, baseline_end, baseline_p_errs,I_0, baselineparams=data_correction(current_voltage_array,baseline_corr_order,vtap_length,springboard_shunt_resistance,trace_baseline_beginning,trace_baseline_end)

'''Perform the power law analysis'''
Ic100, Ic10, n_value, transition, u_100, u_10, nval_err=analysis(baseline_corrected_data,baseline_corr_order,baseline_p_errs,I_0, baselineparams)
#tempave=np.average(temps_array[:,1])
#tempstderr=np.std(temps_array[:,1])/np.sqrt(len(temps_array[:,1]))
#tempstd=np.std(temps_array[:,1])
#temp_polyfit=np.polyfit(baseline_corrected_data[:,0],temps_array[:,1],1)
export_array=np.array(((Ic10,Ic100,n_value)))   #,(tempave,tempstderr,temp_polyfit[0]),(temp_polyfit[1],0,0)))
export_array2=baseline_corrected_data


print('-----------------------------------------------------------')
print('ANALYSIS RESULTS')
print('Ic at 10 muV m-1=', Ic10, '+/-', u_10, 'A')
print('Ic at 100 muV m-1=',Ic100, '+/-', u_100, 'A')
print('nvalue is', n_value, '+/-', nval_err)
print("Board resistance = ", Board_voltage.slope, " +/- ", Board_voltage.stderr, "Ohms")

#print('Average T of Mid. thermometer is:', tempave, '+-', tempstderr)
#print('Linear y=mx+c coeffs for middle temperature are m=',temp_polyfit[0], ',  c=', temp_polyfit[1])
#print('Middle Temp Std Dev =', tempstd)

theo_voltage_array=np.linspace(0.01*vtap_length,0.1*vtap_length,100)
theo_current_array=np.zeros(100)

for i in range(100):
    theo_current_array[i]=Ic100*(theo_voltage_array[i]/(0.1*vtap_length))**(1/n_value)

f = open("0_1T.txt", "a+")
f.write("New trace starts here %\r \r")
for i in range(len(current_voltage_array[:,0])):
    f.write(str(current_voltage_array[i,0]))
    f.write('\t')
    f.write(str(baseline_corrected_data[i,2]))  
    f.write('\r')
f.close()



plt.figure()

plt.subplot(221)
plt.scatter(current_voltage_array[:,0],current_voltage_array[:,1],label='data')
plt.plot(current_voltage_array[:,0],voltage_correction,c='k',label='fit')
plt.xlabel('Current (A)')
plt.ylabel('Voltage (microvolts)')

plt.subplot(222)
plt.scatter(raw_array[:,0],1000*raw_array[:,2])
plt.xlabel('Current (A)')
plt.ylabel('Board Voltage (mV)')
#plt.axvline(tempave,c='k',linewidth=2)
#plt.axvline(tempave+tempstd,c='k',linestyle='--',linewidth=2)
#plt.axvline(tempave-tempstd,c='k',linestyle='--',linewidth=2)


plt.subplot(223)
plt.scatter(current_voltage_array[:,0],baseline_corrected_data[:,2])
plt.xlabel('Current (A)')
plt.ylabel('E-field (uV/m)')
plt.axhline(10, c='k')
plt.axhline(100, c='k')

plt.subplot(224)
plt.scatter(transition[:,0],transition[:,1],label='data')
plt.plot(theo_current_array,theo_voltage_array,c='k',label='fit')
plt.xlabel('Current (A)')
plt.ylabel('Voltage (uV)')
plt.xscale('log')
plt.yscale('log')
plt.ylim(0.008*vtap_length,0.2*vtap_length)
#plt.savefig('B14T60Theta90shunt.svg', format='svg')
#np.savetxt('B14T60Theta2_5E_J_Array.csv', export_array2, delimiter=',')
plt.show()

