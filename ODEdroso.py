# Python script to solve the ODE model applied to Drosophila suzukii and to estimate the parameters
# This file contains all the functions needed
# Created by Luca Rossini
# e-mail: luca.rossini@unitus.it
# Last update: 29 January 2022


# List of import

import pandas as pd
import numpy as np
from Parameters import *
from lmfit import Minimizer, Parameters, report_fit
import scipy.integrate as sc
import scipy.stats as scst
#import multiprocessing as mp


# Define the manager for multiprocessing

#manager = mp.Manager()


# Parameter organization - Real parameters!

    # Rate functions parameters from the file 'Parameters.py'

BriPar_EP_Acquired = a_EP, T_L_EP, T_M_EP, m_EP
BriPar_PA_Acquired = a_PA, T_L_PA, T_M_PA, m_PA
BriPar_Sur_Acquired = a_Sur, T_L_Sur, T_M_Sur, m_Sur

FertPar_Acquired = alpha, gamma, Lambda, delta, tau, T_low, T_max
MortPar_Acquired = k_mort, T_MAX_mort, rho_mort

    # Initial conditions

InitCond_Acquired = E_0, L1_0, L2_0, L3_0, P_0, Am_0, Anmf_0, Amf_0


# Definition of Briere rate function

def BriFunc(time, BriPar, DailyTemp):
    
    try:
        temp = DailyTemp[int(time)]
    except IndexError:
        print('Warning: Skipping IndexError in BriFunc()! \n')
        temp = DailyTemp[0]
    
    if np.any(temp >= BriPar[1]) and np.any(temp <= BriPar[2]):
    
        Bri = BriPar[0] * temp * (temp - BriPar[1]) * pow((BriPar[2] - temp), (1 / BriPar[3]))
    else:
        Bri = 0.00001
    
    return Bri


# Definition of the fertility rate function

def FertFunc(time, FertPar, DailyTemp):

    try:
        temp = DailyTemp[int(time)]
    except IndexError:
        print('Warning: Skipping IndexError in FertFunc()! \n')
        temp = DailyTemp[0]

    if np.any(temp >= FertPar[5]) and np.any(temp <= FertPar[6]):
    
        FP = FertPar[0] * (((FertPar[1] + 1) / (np.pi * (FertPar[2] ** (2 * FertPar[1] + 2)))) * ((FertPar[2] ** 2) - ( ((temp - FertPar[4]) ** 2) + (FertPar[3] ** 2)) ) ** FertPar[1])

    else:
       FP = 0

    return FP


# Definition of the Mortality rate function

def MortFunc(time, MortPar, DailyTemp):
    
    try:
        temp = DailyTemp[int(time)]
    except IndexError:
        print('Warning: Skipping IndexError in MortFunc()! \n')
        temp = DailyTemp[0]

    Survival = MortPar[0] * np.exp(1 + ((MortPar[1] - temp) / MortPar[2]) - np.exp((MortPar[1] - temp) / MortPar[2]))

    MP = 1 - Survival

    if MP <= 0:
        MP = 0
    else:
        MP = 1 - Survival

    return MP


# Definition of the ODE system - odeint function MORTALITY in all the stages!

def Sys_ODE_Full(time, y, BriPar_EP, BriPar_PA, BriPar_Sur, FertPar, MortPar, SexRatio, TempArray):

    E = FertFunc(time, FertPar, TempArray) * y[7] - BriFunc(time, BriPar_EP, TempArray) * y[0] - MortFunc(time, MortPar, TempArray) * y[0]

    L1 = BriFunc(time, BriPar_EP, TempArray) * y[0] - BriFunc(time, BriPar_EP, TempArray) * y[1] - MortFunc(time, MortPar, TempArray) * y[1]

    L2 = BriFunc(time, BriPar_EP, TempArray) * y[1] - BriFunc(time, BriPar_EP, TempArray) * y[2] - MortFunc(time, MortPar, TempArray) * y[2]

    L3 = BriFunc(time, BriPar_EP, TempArray) * y[2] - BriFunc(time, BriPar_EP, TempArray) * y[3] - MortFunc(time, MortPar, TempArray) * y[3]

    P = BriFunc(time, BriPar_EP, TempArray) * y[3] - BriFunc(time, BriPar_PA, TempArray) * y[4] - MortFunc(time, MortPar, TempArray) * y[4]

    Am = (1 - SexRatio) * BriFunc(time, BriPar_PA, TempArray) * y[4] - BriFunc(time, BriPar_Sur, TempArray) * y[5] - MortFunc(time, MortPar, TempArray) * y[5]

    Anmf = SexRatio * BriFunc(time, BriPar_PA, TempArray) * y[4] - y[6]

    Amf = y[6] - BriFunc(time, BriPar_Sur, TempArray) * y[6] - MortFunc(time, MortPar, TempArray) * y[6] - MortFunc(time, MortPar, TempArray) * y[7] - BriFunc(time, BriPar_Sur, TempArray) * y[7]

    Z = np.array([E, L1, L2, L3, P, Am, Anmf, Amf])
    
    return Z


# Definition of the ODE system - odeint function 
# MORTALITY ONLY in selected stages based on life tables

def Sys_ODE(time, y, BriPar_EP, BriPar_PA, BriPar_Sur, FertPar, MortPar, SexRatio, TempArray):

    E = FertFunc(time, FertPar, TempArray) * y[7] - BriFunc(time, BriPar_EP, TempArray) * y[0] - BriFunc(time, BriPar_EP, TempArray) * (MortFunc(time, MortPar, TempArray)) * y[0]

    L1 = BriFunc(time, BriPar_EP, TempArray) * y[0] - BriFunc(time, BriPar_EP, TempArray) * y[1] - BriFunc(time, BriPar_EP, TempArray) * (MortFunc(time, MortPar, TempArray)) * y[1]

    L2 = BriFunc(time, BriPar_EP, TempArray) * y[1] - BriFunc(time, BriPar_EP, TempArray) * y[2] - BriFunc(time, BriPar_EP, TempArray) * (MortFunc(time, MortPar, TempArray)) * y[2]

    L3 = BriFunc(time, BriPar_EP, TempArray) * y[2] - BriFunc(time, BriPar_EP, TempArray) * y[3] - BriFunc(time, BriPar_EP, TempArray) * (MortFunc(time, MortPar, TempArray)) * y[3]

    P = BriFunc(time, BriPar_EP, TempArray) * y[3] - BriFunc(time, BriPar_PA, TempArray) * y[4] - BriFunc(time, BriPar_EP, TempArray) * (MortFunc(time, MortPar, TempArray)) * y[4]

    Am = (1 - SexRatio) * BriFunc(time, BriPar_PA, TempArray) * y[4] - BriFunc(time, BriPar_Sur, TempArray) * y[5] - BriFunc(time, BriPar_EP, TempArray) * MortFunc(time, MortPar, TempArray) * y[5]

    Anmf = SexRatio * BriFunc(time, BriPar_PA, TempArray) * y[4] - y[6]

    Amf = y[6] - BriFunc(time, BriPar_Sur, TempArray) * y[6] - BriFunc(time, BriPar_EP, TempArray) * MortFunc(time, MortPar, TempArray) * y[6] - BriFunc(time, BriPar_EP, TempArray) * MortFunc(time, MortPar, TempArray) * y[7] - BriFunc(time, BriPar_Sur, TempArray) * y[7]

    Z = np.array([E, L1, L2, L3, P, Am, Anmf, Amf])
    
    return Z


# This function feeds the parameters and solves the equation - It returns the ODE solution of the adult males for comparison

# NOTE: if you want to change the stage to compare, change the output of this function!!

def EqSolver(time_Eq, InitCond_Eq, BriPar_EP_Eq, BriPar_PA_Eq, BriPar_Sur_Eq, FertPar_Eq, MortPar_Eq, SR_Eq, Temp_Eq):

    ODE_Sol = sc.odeint(Sys_ODE_Full, y0 = InitCond_Eq, t = time_Eq, args = (BriPar_EP_Eq, BriPar_PA_Eq, BriPar_Sur_Eq, FertPar_Eq, MortPar_Eq, SR_Eq, Temp_Eq,), tfirst = True)

    return ODE_Sol[:, 5]


# This function feeds the parameters and solves the equation - It returns the ODE solution of all the system

def EqSolver_Full(time_Eq, InitCond_Eq, BriPar_EP_Eq, BriPar_PA_Eq, BriPar_Sur_Eq, FertPar_Eq, MortPar_Eq, SR_Eq, Temp_Eq):

    ODE_Sol = sc.odeint(Sys_ODE_Full, y0 = InitCond_Eq, t = time_Eq, args = (BriPar_EP_Eq, BriPar_PA_Eq, BriPar_Sur_Eq, FertPar_Eq, MortPar_Eq, SR_Eq, Temp_Eq,), tfirst = True)

    return ODE_Sol


# This function feeds the parameters and solves the equation using solve_ivp - For comparison purposes!

def EqSolverSolveIVP(time_Eq, InitCond_Eq, BriPar_EP_Eq, BriPar_PA_Eq, BriPar_Sur_Eq, FertPar_Eq, MortPar_Eq, SR_Eq, Temp_Eq):

    t_max = len(time_Eq) - 1

    ODE_Sol = sc.solve_ivp(Sys_ODE, t_span = [0, time_Eq[t_max]],  y0 = InitCond_Eq, args = (BriPar_EP_Eq, BriPar_PA_Eq, BriPar_Sur_Eq, FertPar_Eq, MortPar_Eq, SR_Eq, Temp_Eq,), method = 'DOP853')

    return ODE_Sol
 

# Residual calculator: it is needed by "minimize" from lmfit. Calculate the residuals between the experimental data and the ODE solution

def ResidualCalculator(ParToMinimize, time_Res, InitCond_Res, FertPar_Res, MortPar_Res, SR_Res, Temp_Res, data_observed):

    # Unpack the parameters to minimize and assign them to dedicated variables

        # Briére Egg-Pupa stage

    a0P_EP = ParToMinimize['a_EP'].value
    T_L0P_EP = ParToMinimize['T_L_EP'].value
    T_M0P_EP = ParToMinimize['T_M_EP'].value
    m0P_EP = ParToMinimize['m_EP'].value

    BriPar_Res_EP = [a0P_EP, T_L0P_EP, T_M0P_EP, m0P_EP]

        # Briére Pupa-Adult stage

    a0P_PA = ParToMinimize['a_PA'].value
    T_L0P_PA = ParToMinimize['T_L_PA'].value
    T_M0P_PA = ParToMinimize['T_M_PA'].value
    m0P_PA = ParToMinimize['m_PA'].value

    BriPar_Res_PA = [a0P_PA, T_L0P_PA, T_M0P_PA, m0P_PA]

        # Briére Adult survival

    a0P_Sur = ParToMinimize['a_Sur'].value
    T_L0P_Sur = ParToMinimize['T_L_Sur'].value
    T_M0P_Sur = ParToMinimize['T_M_Sur'].value
    m0P_Sur = ParToMinimize['m_Sur'].value

    BriPar_Res_Sur = [a0P_Sur, T_L0P_Sur, T_M0P_Sur, m0P_Sur]
    
    try:
    
        CalcData = EqSolver(time_Res, InitCond_Res, BriPar_Res_EP, BriPar_Res_PA, BriPar_Res_Sur, FertPar_Res, MortPar_Res, SR_Res, Temp_Res)

    except IndexError:
        print('Warning: Skipping IndexError in ResidualCalculator()! \n')
        CalcData = data_observed + (data_observed * 0.5)
        pass
    
    return CalcData - data_observed


# Function that generates values according to a Normal distribution

def NormalGenerator(InitValues, Iteration):

    # Set the standard deviation as portion of the expected values

    if Iteration == 0:
        Portion = 1

    else:
        Portion = 1 / Iteration
    
    Sigma = np.abs(InitValues * 0.8 * Portion)

    np.random.seed()
    RandNorm = np.random.normal(InitValues, Sigma)

    return RandNorm


# Minimization of the residuals - Estimation of the Briere parameters with least-squares

def LeastSquaresFit(i, ObservedData, time, InitCond_Theorethical, FertPar_Theorethical, MortPar_Theorethical, SR_Theorethical, Temp_Theorethical, MaxIter):
 
    # Definition of the free variables to minimize

    ParToMinimize = Parameters()

        # Briere Egg-Pupa

    ParToMinimize.add('a_EP', value = NormalGenerator(a_MCMC_EP, i), min = 0.5 * pow(10, -4), max = 3.0 * pow(10, -4))
    ParToMinimize.add('T_L_EP', value = NormalGenerator(T_L_MCMC_EP, i), min = 1.0, max = 5.0)
    ParToMinimize.add('T_M_EP', value = NormalGenerator(T_M_MCMC_EP, i), min = 28.0, max = 35.0)
    ParToMinimize.add('m_EP', value = np.abs(NormalGenerator(m_MCMC_EP, i)), min = 3.0, max = 5.0)

        # Briere Pupa-Adult

    ParToMinimize.add('a_PA', value = NormalGenerator(a_MCMC_PA, i), min = 1.5 * pow(10, -4), max = 4.0 * pow(10, -4))
    ParToMinimize.add('T_L_PA', value = NormalGenerator(T_L_MCMC_PA, i), min = 2.0, max = 6.0)
    ParToMinimize.add('T_M_PA', value = NormalGenerator(T_M_MCMC_PA, i), min = 30.0, max = 35.0)
    ParToMinimize.add('m_PA', value = np.abs(NormalGenerator(m_MCMC_PA, i)), min = 3.0, max = 5.0)

        # Briere Adult survival

    ParToMinimize.add('a_Sur', value = NormalGenerator(a_MCMC_Sur, i), min = 3.0 * pow(10, -5), max = 1.0 * pow(10, -4))
    ParToMinimize.add('T_L_Sur', value = NormalGenerator(T_L_MCMC_Sur, i), min = -6.0, max =-1.0)
    ParToMinimize.add('T_M_Sur', value = NormalGenerator(T_M_MCMC_Sur, i), min = 28.0, max =31.0)
    ParToMinimize.add('m_Sur', value = np.abs(NormalGenerator(m_MCMC_Sur, i)), min = 1.5, max =3.5)

    # Fitting - LeastSquares

    FitLstSq = Minimizer(ResidualCalculator, ParToMinimize, fcn_args = (time, InitCond_Theorethical, FertPar_Theorethical, MortPar_Theorethical, SR_Theorethical, Temp_Theorethical, ObservedData))

    FitResults = FitLstSq.minimize(method='leastsq', max_nfev=MaxIter)

    # Provides the best fit numerical function

    FinalValues = ObservedData + FitResults.residual.reshape(ObservedData.shape)

    # Temporary dictionary summarizing the iteration scores

    ChiSquareTemporary = {'a_EP': FitResults.params['a_EP'].value, 
                          'a_Err_EP': FitResults.params['a_EP'].stderr, 
                          'T_L_EP': FitResults.params['T_L_EP'].value, 
                          'T_L_Err_EP': FitResults.params['T_L_EP'].stderr, 
                          'T_M_EP': FitResults.params['T_M_EP'].value, 
                          'T_M_Err_EP': FitResults.params['T_M_EP'].stderr, 
                          'm_EP': FitResults.params['m_EP'].value, 
                          'm_Err_EP': FitResults.params['m_EP'].stderr,
                          'a_PA': FitResults.params['a_PA'].value, 
                          'a_Err_PA': FitResults.params['a_PA'].stderr, 
                          'T_L_PA': FitResults.params['T_L_PA'].value, 
                          'T_L_Err_PA': FitResults.params['T_L_PA'].stderr, 
                          'T_M_PA': FitResults.params['T_M_PA'].value, 
                          'T_M_Err_PA': FitResults.params['T_M_PA'].stderr, 
                          'm_PA': FitResults.params['m_PA'].value, 
                          'm_Err_PA': FitResults.params['m_PA'].stderr,
                          'a_Sur': FitResults.params['a_Sur'].value, 
                          'a_Err_Sur': FitResults.params['a_Sur'].stderr, 
                          'T_L_Sur': FitResults.params['T_L_Sur'].value, 
                          'T_L_Err_Sur': FitResults.params['T_L_Sur'].stderr, 
                          'T_M_Sur': FitResults.params['T_M_Sur'].value, 
                          'T_M_Err_Sur': FitResults.params['T_M_Sur'].stderr, 
                          'm_Sur': FitResults.params['m_Sur'].value, 
                          'm_Err_Sur': FitResults.params['m_Sur'].stderr, 
                          'ChiSquare': FitResults.chisqr,
                          'FinalValues': FinalValues,
                          'FitResults': FitResults}

    # Return the row to add to update the for loop in the main code

    return ChiSquareTemporary


# Add a string to shared dictionary in parallel process for the least squares fit step

def DictString(LeastSquare_DataFrame, i, yobs, day_obs, InitCond_Acquired, FertPar_Acquired, MortPar_Acquired, SR, Temp, MaxIterations_LeastSq):

    StringName = 'Iteration_{}'.format(i)

    if StringName not in LeastSquare_DataFrame:

        LeastSquare_DataFrame[StringName] = {}

    LeastSquare_DataFrame[StringName] = LeastSquaresFit(i, yobs, day_obs, InitCond_Acquired, FertPar_Acquired, MortPar_Acquired, SR, Temp, MaxIterations_LeastSq)


# Genetic algorithm to optimize the Least Squares, the step 1 of the MCMC process

def GeneticAlgorithm_LSF(BestValues_DataFrame, SharedDict_LSF, MaxEstimations_ForLoop, yobs, day_obs, InitCond_Acquired, FertPar_Acquired, MortPar_Acquired, SR, Temp, MaxIterations_LeastSq):

        # Define the new variable MaxEstimations_LeastSq - It takes only the first quarter of the data for the novel estimation:

    MaxEstimations_GenAlgo = np.int64((MaxEstimations_ForLoop) / 4)

        # First loop: calculates LeastSquaresFit() and saves its output

    Procs_GenAlgo = list()

    for processes in range(MaxEstimations_GenAlgo):

        for subiter in range(4):
        
            # New initial conditions from BestValues_DataFrame
            # From this one it estimates three random values!
            
                # Brière Egg-Pupa

            a_MCMC_EP = BestValues_DataFrame['a_EP'][processes]
            T_L_MCMC_EP = BestValues_DataFrame['T_L_EP'][processes]
            T_M_MCMC_EP = BestValues_DataFrame['T_M_EP'][processes]
            m_MCMC_EP = BestValues_DataFrame['m_EP'][processes]

                # Brière Pupa-Adult

            a_MCMC_PA = BestValues_DataFrame['a_PA'][processes]
            T_L_MCMC_PA = BestValues_DataFrame['T_L_PA'][processes]
            T_M_MCMC_PA = BestValues_DataFrame['T_M_PA'][processes]
            m_MCMC_PA = BestValues_DataFrame['m_PA'][processes]

                # Brière Adult Survival

            a_MCMC_Sur = BestValues_DataFrame['a_Sur'][processes]
            T_L_MCMC_Sur = BestValues_DataFrame['T_L_Sur'][processes]
            T_M_MCMC_Sur = BestValues_DataFrame['T_M_Sur'][processes]
            m_MCMC_Sur = BestValues_DataFrame['m_Sur'][processes]
            
            # Continues to add the keys into the shared dictionary
            # Sequential numbering of iterations name: step necessary because the enumeration
            # starts from 0 and not from 1 as expected

            if processes == 0 and subiter == 0:

                NewStringValue = MaxEstimations_ForLoop + processes + subiter

            elif processes == 0 and subiter != 0:

                NewStringValue = MaxEstimations_ForLoop + processes + subiter

            else:
        
                NewStringValue = MaxEstimations_ForLoop + processes + subiter + (3 * processes)

            # String numbering consequential after the previous update

            StringName ='Iteration_{}'.format(NewStringValue)
            SharedDict_LSF[StringName] = manager.dict()

            # Creates the parallel processes

            MultiProcess_FitLSQ = mp.Process(target = DictString, args = (SharedDict_LSF, NewStringValue, yobs, day_obs, InitCond_Acquired, FertPar_Acquired, MortPar_Acquired, SR, Temp, MaxIterations_LeastSq))
    
            # Stores the processes and starts them

            Procs_GenAlgo.append(MultiProcess_FitLSQ)
            MultiProcess_FitLSQ.start()
  
            print('\tGenetic Algorithm - LeastSquares Fit step: Iteration', NewStringValue, '\n')


        # Second loop: it joins the processes

    for MultiProcess_FitLSQ in Procs_GenAlgo:
        MultiProcess_FitLSQ.join()

        # Total number of iteration in the LSF part:
    
    TotIter_LSF = MaxEstimations_ForLoop + (MaxEstimations_GenAlgo * 4)

    return TotIter_LSF, SharedDict_LSF


# Function that calculates log-probability for the MCMC step

def LogProbabilityCalculator_MCMC(SimulatedData, ObservedData):
    
    Length = np.int64(len(ObservedData))
    TotalProbability = []

    for i in range(Length):

        MassProbability = scst.poisson.pmf(np.int64(ObservedData[i]), SimulatedData[i])
        
        if MassProbability > 1 * pow(10, -30):
            Log_MassProbability = np.log(MassProbability)
        else:
            Log_MassProbability = np.log(1 * pow(10, -30))
        
        # Append to the storage array

        TotalProbability = np.append(TotalProbability, Log_MassProbability)
    
    LogProbability = sum(TotalProbability)

    return LogProbability


# Function that calculates the single iteration of the MCMC step

def SingleIteration_ChainCalculator(InputData):
    
    # Calculate the length of the columns

    DictColumn_Length = np.int64(len(InputData['a_EP']))

    # Select a random row from the best values LSF
    
    np.random.seed()
    RandomItem_FromBestLSF = np.random.randint(low = 0, high = DictColumn_Length)
    
    # Generate random parameter values from the previously selected value

        # Briere Egg-Pupa

    a_EP = NormalGenerator(InputData['a_EP'][RandomItem_FromBestLSF], 0.15)
    T_L_EP = NormalGenerator(InputData['T_L_EP'][RandomItem_FromBestLSF], 0.15)
    T_M_EP = NormalGenerator(InputData['T_M_EP'][RandomItem_FromBestLSF], 0.15)
    m_EP = np.abs(NormalGenerator(InputData['m_EP'][RandomItem_FromBestLSF], 0.15))

    BriPar_RandomMCMC_EP = [a_EP, T_L_EP, T_M_EP, m_EP]

        # Briere Pupa-Adult

    a_PA = NormalGenerator(InputData['a_PA'][RandomItem_FromBestLSF], 0.15)
    T_L_PA = NormalGenerator(InputData['T_L_PA'][RandomItem_FromBestLSF], 0.15)
    T_M_PA = NormalGenerator(InputData['T_M_PA'][RandomItem_FromBestLSF], 0.15)
    m_PA = np.abs(NormalGenerator(InputData['m_PA'][RandomItem_FromBestLSF], 0.15))

    BriPar_RandomMCMC_PA = [a_PA, T_L_PA, T_M_PA, m_PA]

        # Briere Adult Survival

    a_Sur = NormalGenerator(InputData['a_Sur'][RandomItem_FromBestLSF], 0.15)
    T_L_Sur = NormalGenerator(InputData['T_L_Sur'][RandomItem_FromBestLSF], 0.15)
    T_M_Sur = NormalGenerator(InputData['T_M_Sur'][RandomItem_FromBestLSF], 0.15)
    m_Sur = np.abs(NormalGenerator(InputData['m_Sur'][RandomItem_FromBestLSF], 0.15))

    BriPar_RandomMCMC_Sur = [a_Sur, T_L_Sur, T_M_Sur, m_Sur]

    # Solve the ODE system with the random values

    EqSolution = EqSolver(InputData['Time'], InputData['InitialConditions'], BriPar_RandomMCMC_EP, BriPar_RandomMCMC_PA, BriPar_RandomMCMC_Sur, InputData['FertParameters'], InputData['MortParameters'], InputData['SexRatio'], InputData['Temperatures'])

    # Calculate the log-probability

    LogProbability = LogProbabilityCalculator_MCMC(EqSolution, InputData['ObservedData'])

    return {'a_EP': a_EP, 'T_L_EP': T_L_EP, 'T_M_EP': T_M_EP, 'm_EP': m_EP, 'a_PA': a_PA, 'T_L_PA': T_L_PA, 'T_M_PA': T_M_PA, 'm_PA': m_PA, 'a_Sur': a_Sur, 'T_L_Sur': T_L_Sur, 'T_M_Sur': T_M_Sur, 'm_Sur': m_Sur, 'LogProbability': LogProbability}


# Function that calculates the iterations into the chain

def ChainCalculator(InputData):

    # Define the maximum number of iterations per chain, adding 100 more iterations for tuning

    Max = np.int64(InputData['MaxIter_PerChain']) + 101

    # Arrays to store the values

        # Briere Egg-Pupa

    a_EP = []
    T_L_EP = []
    T_M_EP = []
    m_EP = []

        # Briere Pupa-Adult

    a_PA = []
    T_L_PA = []
    T_M_PA = []
    m_PA = []

        # Briere Adult Survival

    a_Sur = []
    T_L_Sur = []
    T_M_Sur = []
    m_Sur = []

        # Likelihood

    Like = []

    # For loop for calculations

    for i in range(Max):

        TempDict = SingleIteration_ChainCalculator(InputData)

        if i == 0:
        
                # Briere Egg-Pupa

            a_EP = [TempDict['a_EP']]
            T_L_EP = [TempDict['T_L_EP']]
            T_M_EP = [TempDict['T_M_EP']]
            m_EP = [TempDict['m_EP']]

                # Briere Pupa-Adult

            a_PA = [TempDict['a_PA']]
            T_L_PA = [TempDict['T_L_PA']]
            T_M_PA = [TempDict['T_M_PA']]
            m_PA = [TempDict['m_PA']]

                # Briere Adult Survival

            a_Sur = [TempDict['a_Sur']]
            T_L_Sur = [TempDict['T_L_Sur']]
            T_M_Sur = [TempDict['T_M_Sur']]
            m_Sur = [TempDict['m_Sur']]

            Like = [TempDict['LogProbability']]
            Previous_Like = TempDict['LogProbability']
            Current_Like = TempDict['LogProbability']
            
        else:

            TempDict = SingleIteration_ChainCalculator(InputData)
            Current_Like = TempDict['LogProbability']
            Previous_Like = Like[i-1]

        if  Current_Like > Previous_Like:

                # Briére Egg-Pupa
        
            a_EP = np.append(a_EP, TempDict['a_EP'])
            T_L_EP = np.append(T_L_EP, TempDict['T_L_EP'])
            T_M_EP = np.append(T_M_EP, TempDict['T_M_EP'])
            m_EP = np.append(m_EP, TempDict['m_EP'])

                # Briére Pupa-Adult

            a_PA = np.append(a_PA, TempDict['a_PA'])
            T_L_PA = np.append(T_L_PA, TempDict['T_L_PA'])
            T_M_PA = np.append(T_M_PA, TempDict['T_M_PA'])
            m_PA = np.append(m_PA, TempDict['m_PA'])

                # Briere Adult Survival

            a_Sur = np.append(a_Sur, TempDict['a_Sur'])
            T_L_Sur = np.append(T_L_Sur, TempDict['T_L_Sur'])
            T_M_Sur = np.append(T_M_Sur, TempDict['T_M_Sur'])
            m_Sur = np.append(m_Sur, TempDict['m_Sur'])

                # Likelihood

            Like = np.append(Like, TempDict['LogProbability'])
        
        else:

                # Briére Egg-Pupa

            a_EP = np.append(a_EP, a_EP[i-1])
            T_L_EP = np.append(T_L_EP, T_L_EP[i-1])
            T_M_EP = np.append(T_M_EP, T_M_EP[i-1])
            m_EP = np.append(m_EP, m_EP[i-1])

                # Briére Pupa-Adult

            a_PA = np.append(a_PA, a_PA[i-1])
            T_L_PA = np.append(T_L_PA, T_L_PA[i-1])
            T_M_PA = np.append(T_M_PA, T_M_PA[i-1])
            m_PA = np.append(m_PA, m_PA[i-1])

                # Briére Adult Survival

            a_Sur = np.append(a_Sur, a_Sur[i-1])
            T_L_Sur = np.append(T_L_Sur, T_L_Sur[i-1])
            T_M_Sur = np.append(T_M_Sur, T_M_Sur[i-1])
            m_Sur = np.append(m_Sur, m_Sur[i-1])

                # Likelihood

            Like = np.append(Like, Like[i-1])

    # Report the trace but it burns the first 100 iterations (more reliable result at the end)

    Trace = {'a_EP': a_EP[99: Max],
             'T_L_EP': T_L_EP[99: Max],
             'T_M_EP': T_M_EP[99: Max],
             'm_EP': m_EP[99: Max],
             'a_PA': a_PA[99: Max],
             'T_L_PA': T_L_PA[99: Max],
             'T_M_PA': T_M_PA[99: Max],
             'm_PA': m_PA[99: Max],
             'a_Sur': a_Sur[99: Max],
             'T_L_Sur': T_L_Sur[99: Max],
             'T_M_Sur': T_M_Sur[99: Max],
             'm_Sur': m_Sur[99: Max],
             'LogProbability': Like[99: Max]}
    
    return Trace


# Function to calculate the chains in parallel using Ray Pool.starmap()

def ChainRepeater(InputDict, Chains):

    TraceResult = {}

    StringName = 'Chain_{}'.format(Chains)

    print('\tMonteCarlo Chain number:', Chains, '\n')

    if StringName not in TraceResult:

        TraceResult[StringName] = {}

    TraceResult[StringName] = ChainCalculator(InputDict)

    return TraceResult




