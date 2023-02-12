
# Main file of the ODE system solution. Launch this script to solve the system and
# plot the results.
# Created by Luca Rossini
# e-mail: luca.rossini@unitus.it
# Last update: 23 November 2022


#            ----- Preliminary notes ------

# 1. Check with 'ulimit -n 4096' how many files your system can open simultaneously

# 2. Start the Ray utilities:

#    Stop firewall on each node: sudo systemctl stop firewalld

#    Head node (Bender): ray start --head --num-cpus=96 --redis-password='Password'

#    Worker node (Marvin): ray start --address=192.4.21.21:6379 --num-cpus=96 --redis-password='Password'

# Remember to reactivate the firewall after the script usage!!



# ----- Beginning of the Script ----- 

# List of import

from Parameters import *
from ODEdroso import *
import pandas as pd
import scipy.optimize as op

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from lmfit import report_fit, fit_report

import multiprocessing as mp
import time
from datetime import datetime
import ray
from ray.job_config import JobConfig
import ray.util.multiprocessing.pool as rmp

from itertools import repeat

# Start the Ray cluster manager

ray.init(address='192.4.21.21:6379', _redis_password='Password', job_config = JobConfig(runtime_env={"working_dir": "."}))


# Start the chronometer to retrieve the total time of execution

StartTime_LSF = time.time()


# Acquisition of the observed data to make the MCMC possible

data_exp = pd.read_csv("Experimental_data-Campo1.csv", sep=",", header=0)
data_exp.columns = ["Day", "Temperature", "Ad_male"]

# Interpolation of the missing data to make the experimental data comparable with simulations

data_exp = data_exp.replace('', np.nan)
data_exp = data_exp.interpolate()

# Retrieve the time vector from experimental data

day_ob = data_exp["Day"]
day_obs = day_ob.to_numpy()

# Retrieve the daily temperature array

Temp_data = data_exp['Temperature']
Temp = Temp_data.to_numpy()

# Retrieve monitoring data

yob = data_exp['Ad_male']
yobs = yob.to_numpy() 




# First step: individuates good initial values for the MCMC by using the least squares method
# Those values are stored into a nested dictionary, and processed in the next steps of the program

# This step is called as "Pre Evaluation - Least Squares Fit step"

    # Define the list of processes (Procs)

Procs = list()
print('\n') # Just for shell visualization


    # Define the dictionary to store the LeastSq results and the ChiSquare list to reorder

SharedDict_LSF = manager.dict()


    # First loop: calculates LeastSquaresFit() and saves its output

for processes in range(MaxEstimations_LeastSq):

    # Creates the keys into the shared dictionary

    StringName ='Iteration_{}'.format(processes)
    SharedDict_LSF[StringName] = manager.dict()

    # Creates the parallel processes

    MultiProcess_FitLSQ = mp.Process(target = DictString, args = (SharedDict_LSF, processes, yobs, day_obs, InitCond_Acquired, FertPar_Acquired, MortPar_Acquired, SR, Temp, MaxIterations_LeastSq))
    
    # Stores the processes and starts them

    Procs.append(MultiProcess_FitLSQ)
    MultiProcess_FitLSQ.start()
  
    print('\tPre Evaluation - LeastSquares Fit step: Iteration', processes, '\n')


    # Second loop: it joins the processes

for MultiProcess_FitLSQ in Procs:
    MultiProcess_FitLSQ.join()

    # Save the least squares best fit values, their errors, the numerical solution and the minimizer outputs in a data frame

BestValues_DataFrame = pd.DataFrame.from_dict(SharedDict_LSF, orient='index')

    # Reorder the dataframe by ChiSquare column in ascending order

BestValues_DataFrame = BestValues_DataFrame.sort_values(by = ['ChiSquare'])


# Second step: first optimization with a genetic algorithm using least squares minimization
# This step is called as "Genetic Algorithm - Least Squares Fit step"


    # First iteration of Genetic Algorithm

TotIter_LSF, SharedDict_LSF = GeneticAlgorithm_LSF(BestValues_DataFrame, SharedDict_LSF, MaxEstimations_LeastSq, yobs, day_obs, InitCond_Acquired, FertPar_Acquired, MortPar_Acquired, SR, Temp, MaxIterations_LeastSq)

    # Update the dataframe previously filled in the Pre Evaluation step

BestValues_DataFrame = pd.DataFrame.from_dict(SharedDict_LSF, orient='index')

    # Reorder the dataframe for the second step

BestValues_DataFrame = BestValues_DataFrame.sort_values(by = ['ChiSquare'])


    # Additional iterations of Genetic Algorithm, according to the
    # MaxIterations_GeneticAlgorithm variable in Parameters.py

for i in range(MaxIterations_GeneticAlgorithm):

    print('\tGenetic Algorithm - Foor loop number', i, '\n')

    TotIter_LSF, SharedDict_LSF = GeneticAlgorithm_LSF(BestValues_DataFrame, SharedDict_LSF, TotIter_LSF, yobs, day_obs, InitCond_Acquired, FertPar_Acquired, MortPar_Acquired, SR, Temp, MaxIterations_LeastSq)

    BestValues_DataFrame = pd.DataFrame.from_dict(SharedDict_LSF, orient='index')
    BestValues_DataFrame = BestValues_DataFrame.sort_values(by = ['ChiSquare'])


    # Save the reordered data frame in a .cvs file

BestValues_DataFrame.drop(columns=['FinalValues', 'FitResults']).to_csv('LeastSquares-BestValues.csv', sep=',')

    # Get the end time of the LSF step

EndTime_LSF = time.time()


    # Retrieve the name of the best iteration (first row, first element)

BestIterationName = BestValues_DataFrame.index[0]


    # Print the values of the best LSF on terminal

print('\t----------------------\n')
print('Least Squares step FitResults: \n')
ReportLSF = report_fit(BestValues_DataFrame['FitResults'][0])
print('\nBest Iteration name is: ', BestIterationName)
print('\nTotal number of iterations in LSF step: ', TotIter_LSF) 

    # Print the time of execution

print('\nTotal time of execution of the LSF step:', round(EndTime_LSF - StartTime_LSF, 3), 'seconds\n')


# Average and standard deviation of the first "MaxEstimations_LeastSq" values belonging to BestValues_DataFrame

    # Briére Egg-Pupa

Avg_a_from_step_1_EP = np.mean(BestValues_DataFrame['a_EP'][0:99])
DevSt_a_from_step_1_EP = np.std(BestValues_DataFrame['a_EP'][0:99])

Avg_T_L_from_step_1_EP = np.mean(BestValues_DataFrame['T_L_EP'][0:99])
DevSt_T_L_from_step_1_EP = np.std(BestValues_DataFrame['T_L_EP'][0:99])

Avg_T_M_from_step_1_EP = np.mean(BestValues_DataFrame['T_M_EP'][0:99])
DevSt_T_M_from_step_1_EP = np.std(BestValues_DataFrame['T_M_EP'][0:99])

Avg_m_from_step_1_EP = np.mean(BestValues_DataFrame['m_EP'][0:99])
DevSt_m_from_step_1_EP = np.std(BestValues_DataFrame['m_EP'][0:99])

    # Briére Pupa-Adult

Avg_a_from_step_1_PA = np.mean(BestValues_DataFrame['a_PA'][0:99])
DevSt_a_from_step_1_PA = np.std(BestValues_DataFrame['a_PA'][0:99])

Avg_T_L_from_step_1_PA = np.mean(BestValues_DataFrame['T_L_PA'][0:99])
DevSt_T_L_from_step_1_PA = np.std(BestValues_DataFrame['T_L_PA'][0:99])

Avg_T_M_from_step_1_PA = np.mean(BestValues_DataFrame['T_M_PA'][0:99])
DevSt_T_M_from_step_1_PA = np.std(BestValues_DataFrame['T_M_PA'][0:99])

Avg_m_from_step_1_PA = np.mean(BestValues_DataFrame['m_PA'][0:99])
DevSt_m_from_step_1_PA = np.std(BestValues_DataFrame['m_PA'][0:99])

    # Briére Adult Survival

Avg_a_from_step_1_Sur = np.mean(BestValues_DataFrame['a_Sur'][0:99])
DevSt_a_from_step_1_Sur = np.std(BestValues_DataFrame['a_Sur'][0:99])

Avg_T_L_from_step_1_Sur = np.mean(BestValues_DataFrame['T_L_Sur'][0:99])
DevSt_T_L_from_step_1_Sur = np.std(BestValues_DataFrame['T_L_Sur'][0:99])

Avg_T_M_from_step_1_Sur = np.mean(BestValues_DataFrame['T_M_Sur'][0:99])
DevSt_T_M_from_step_1_Sur = np.std(BestValues_DataFrame['T_M_Sur'][0:99])

Avg_m_from_step_1_Sur = np.mean(BestValues_DataFrame['m_Sur'][0:99])
DevSt_m_from_step_1_Sur = np.std(BestValues_DataFrame['m_Sur'][0:99])


    # Print the statistics of LSF step on terminal

print('\t----------------------\n')
print('Statistics from LSF step (only first 100 best iterations considered): \n')
print('\tParameter: mean +/- standard deviation \n')
print('\ta_EP:    ', Avg_a_from_step_1_EP, '+/-', DevSt_a_from_step_1_EP, '\n')
print('\tT_L_EP:  ', Avg_T_L_from_step_1_EP, '+/-', DevSt_T_L_from_step_1_EP, '\n')
print('\tT_M_EP:  ', Avg_T_M_from_step_1_EP, '+/-', DevSt_T_M_from_step_1_EP, '\n')
print('\tm_EP:    ', Avg_m_from_step_1_EP, '+/-', DevSt_m_from_step_1_EP, '\n')
print('\n')
print('\ta_PA:    ', Avg_a_from_step_1_PA, '+/-', DevSt_a_from_step_1_PA, '\n')
print('\tT_L_PA:  ', Avg_T_L_from_step_1_PA, '+/-', DevSt_T_L_from_step_1_PA, '\n')
print('\tT_M_PA:  ', Avg_T_M_from_step_1_PA, '+/-', DevSt_T_M_from_step_1_PA, '\n')
print('\tm_PA:    ', Avg_m_from_step_1_PA, '+/-', DevSt_m_from_step_1_PA, '\n')
print('\n')
print('\ta_Sur:    ', Avg_a_from_step_1_Sur, '+/-', DevSt_a_from_step_1_Sur, '\n')
print('\tT_L_Sur:  ', Avg_T_L_from_step_1_Sur, '+/-', DevSt_T_L_from_step_1_Sur, '\n')
print('\tT_M_Sur:  ', Avg_T_M_from_step_1_Sur, '+/-', DevSt_T_M_from_step_1_Sur, '\n')
print('\tm_Sur:    ', Avg_m_from_step_1_Sur, '+/-', DevSt_m_from_step_1_Sur, '\n')
print('\t----------------------\n')




# Third step: run the MCMC algorithm
# Those values are stored into a nested dictionary, and processed in the next steps of the program

    # Start the calculation of the time for the report

StartTime_MCMC = time.time()

    # Start the Ray pool process to work on the cluster

pool = rmp.Pool()


    # Preparing the input dictionary for the MCMC step

ValuesFromLSF_ToConsider = np.int64(TotIter_LSF / MaxEstimations_LeastSq)

print('Values to consider for MCMC: ', ValuesFromLSF_ToConsider, '\n')

InputDict_MCMC = {'MaxIter_PerChain': NumberIterations_PerChain_MCMC,
                  'ObservedData': yobs,
                  'a_EP': BestValues_DataFrame['a_EP'].values[0: ValuesFromLSF_ToConsider],
                  'T_L_EP': BestValues_DataFrame['T_L_EP'].values[0: ValuesFromLSF_ToConsider],
                  'T_M_EP': BestValues_DataFrame['T_M_EP'].values[0: ValuesFromLSF_ToConsider],
                  'm_EP': BestValues_DataFrame['m_EP'].values[0: ValuesFromLSF_ToConsider],
                  'a_PA': BestValues_DataFrame['a_PA'].values[0: ValuesFromLSF_ToConsider],
                  'T_L_PA': BestValues_DataFrame['T_L_PA'].values[0: ValuesFromLSF_ToConsider],
                  'T_M_PA': BestValues_DataFrame['T_M_PA'].values[0: ValuesFromLSF_ToConsider],
                  'm_PA': BestValues_DataFrame['m_PA'].values[0: ValuesFromLSF_ToConsider],
                  'a_Sur': BestValues_DataFrame['a_Sur'].values[0: ValuesFromLSF_ToConsider],
                  'T_L_Sur': BestValues_DataFrame['T_L_Sur'].values[0: ValuesFromLSF_ToConsider],
                  'T_M_Sur': BestValues_DataFrame['T_M_Sur'].values[0: ValuesFromLSF_ToConsider],
                  'm_Sur': BestValues_DataFrame['m_Sur'].values[0: ValuesFromLSF_ToConsider],
                  'Time': day_obs,
                  'InitialConditions': InitCond_Acquired,
                  'FertParameters': FertPar_Acquired,
                  'MortParameters': MortPar_Acquired,
                  'SexRatio': SR,
                  'Temperatures': Temp}


IteratorChains_MCMC = np.int64([i for i in range(NumberChains_MCMC)])


# Execute the MCMC chains with Ray pool actors

Traces = pool.starmap(ChainRepeater, list(zip(repeat(InputDict_MCMC), IteratorChains_MCMC)))


# Define the dataframe to collect the traces

ChainsAndTraces = {}


# Retrieves the trace values from the chains and order them in a dataframe

for i in Traces:

    ChainsAndTraces.update(i)


# Store the traces to a dataframe that will be subsequently reordered to get results

ChainsAndTraces = pd.DataFrame.from_dict(ChainsAndTraces, orient='index')


# Preparing the traces for each parameter

    # Briere Egg-Pupa

Traces_a_EP = {}
Traces_T_L_EP = {}
Traces_T_M_EP = {}
Traces_m_EP = {}

    # Briére Pupa-Adult

Traces_a_PA = {}
Traces_T_L_PA = {}
Traces_T_M_PA = {}
Traces_m_PA = {}

    # Briére Adult Survival

Traces_a_Sur = {}
Traces_T_L_Sur = {}
Traces_T_M_Sur = {}
Traces_m_Sur = {}

    # Traces

Traces_LogLikelihood = {}

# Extrapolate and reorder the traces

for j in range(np.int64(NumberChains_MCMC)):

    for i in range(np.int64(NumberIterations_PerChain_MCMC)):

        ChainName = 'Chain_{}'.format(j)
        IterationName = 'Iteration_{}'.format(i)

            # Briére Egg-Pupa

        if ChainName not in Traces_a_EP:

            Traces_a_EP[ChainName] = {}
 
        Traces_a_EP[ChainName].update({IterationName: ChainsAndTraces['a_EP'][j][i]})

        if ChainName not in Traces_T_L_EP:

            Traces_T_L_EP[ChainName] = {}

        Traces_T_L_EP[ChainName].update({IterationName: ChainsAndTraces['T_L_EP'][j][i]})

        if ChainName not in Traces_T_M_EP:

            Traces_T_M_EP[ChainName] = {}

        Traces_T_M_EP[ChainName].update({IterationName: ChainsAndTraces['T_M_EP'][j][i]})

        if ChainName not in Traces_m_EP:

            Traces_m_EP[ChainName] = {}

        Traces_m_EP[ChainName].update({IterationName: ChainsAndTraces['m_EP'][j][i]})

            # Briére Pupa-Adult

        if ChainName not in Traces_a_PA:

            Traces_a_PA[ChainName] = {}
 
        Traces_a_PA[ChainName].update({IterationName: ChainsAndTraces['a_PA'][j][i]})

        if ChainName not in Traces_T_L_PA:

            Traces_T_L_PA[ChainName] = {}

        Traces_T_L_PA[ChainName].update({IterationName: ChainsAndTraces['T_L_PA'][j][i]})

        if ChainName not in Traces_T_M_PA:

            Traces_T_M_PA[ChainName] = {}

        Traces_T_M_PA[ChainName].update({IterationName: ChainsAndTraces['T_M_PA'][j][i]})

        if ChainName not in Traces_m_PA:

            Traces_m_PA[ChainName] = {}

        Traces_m_PA[ChainName].update({IterationName: ChainsAndTraces['m_PA'][j][i]})

            # Briére Adult Survival

        if ChainName not in Traces_a_Sur:

            Traces_a_Sur[ChainName] = {}
 
        Traces_a_Sur[ChainName].update({IterationName: ChainsAndTraces['a_Sur'][j][i]})

        if ChainName not in Traces_T_L_Sur:

            Traces_T_L_Sur[ChainName] = {}

        Traces_T_L_Sur[ChainName].update({IterationName: ChainsAndTraces['T_L_Sur'][j][i]})

        if ChainName not in Traces_T_M_Sur:

            Traces_T_M_Sur[ChainName] = {}

        Traces_T_M_Sur[ChainName].update({IterationName: ChainsAndTraces['T_M_Sur'][j][i]})

        if ChainName not in Traces_m_Sur:

            Traces_m_Sur[ChainName] = {}

        Traces_m_Sur[ChainName].update({IterationName: ChainsAndTraces['m_Sur'][j][i]})

            # Log likelihood

        if ChainName not in Traces_LogLikelihood:

            Traces_LogLikelihood[ChainName] = {}

        Traces_LogLikelihood[ChainName].update({IterationName: ChainsAndTraces['LogProbability'][j][i]})


# Final step to save traces from each chain for each parameter

    # Briére Egg-Pupa

Traces_a_EP = pd.DataFrame.from_dict(Traces_a_EP, orient='columns')
Traces_T_L_EP = pd.DataFrame.from_dict(Traces_T_L_EP, orient='columns')
Traces_T_M_EP = pd.DataFrame.from_dict(Traces_T_M_EP, orient='columns')
Traces_m_EP = pd.DataFrame.from_dict(Traces_m_EP, orient='columns')

    # Briére Pupa-Adult

Traces_a_PA = pd.DataFrame.from_dict(Traces_a_PA, orient='columns')
Traces_T_L_PA = pd.DataFrame.from_dict(Traces_T_L_PA, orient='columns')
Traces_T_M_PA = pd.DataFrame.from_dict(Traces_T_M_PA, orient='columns')
Traces_m_PA = pd.DataFrame.from_dict(Traces_m_PA, orient='columns')

    # Briére Adult Survival

Traces_a_Sur = pd.DataFrame.from_dict(Traces_a_Sur, orient='columns')
Traces_T_L_Sur = pd.DataFrame.from_dict(Traces_T_L_Sur, orient='columns')
Traces_T_M_Sur = pd.DataFrame.from_dict(Traces_T_M_Sur, orient='columns')
Traces_m_Sur = pd.DataFrame.from_dict(Traces_m_Sur, orient='columns')

    # Log likelihood

Traces_LogLikelihood = pd.DataFrame.from_dict(Traces_LogLikelihood, orient='columns')


# Save the traces in dedicated files

    # Briére Egg-Pupa

Traces_a_EP.to_csv('./MCMC-Res/Trace_a_EP-MCMC.csv', sep=';')
Traces_T_L_EP.to_csv('./MCMC-Res/Trace_T_L_EP-MCMC.csv', sep=';')
Traces_T_M_EP.to_csv('./MCMC-Res/Trace_T_M_EP-MCMC.csv', sep=';')
Traces_m_EP.to_csv('./MCMC-Res/Trace_m_EP-MCMC.csv', sep=';')

    # Briere Pupa-Adult

Traces_a_PA.to_csv('./MCMC-Res/Trace_a_PA-MCMC.csv', sep=';')
Traces_T_L_PA.to_csv('./MCMC-Res/Trace_T_L_PA-MCMC.csv', sep=';')
Traces_T_M_PA.to_csv('./MCMC-Res/Trace_T_M_PA-MCMC.csv', sep=';')
Traces_m_PA.to_csv('./MCMC-Res/Trace_m_PA-MCMC.csv', sep=';')

    # Briére Adult Survival

Traces_a_Sur.to_csv('./MCMC-Res/Trace_a_Sur-MCMC.csv', sep=';')
Traces_T_L_Sur.to_csv('./MCMC-Res/Trace_T_L_Sur-MCMC.csv', sep=';')
Traces_T_M_Sur.to_csv('./MCMC-Res/Trace_T_M_Sur-MCMC.csv', sep=';')
Traces_m_Sur.to_csv('./MCMC-Res/Trace_m_Sur-MCMC.csv', sep=';')

Traces_LogLikelihood.to_csv('./MCMC-Res/Trace_LogLike-MCMC.csv', sep=';')


    # Stop the calculation of the time for the report

EndTime_MCMC = time.time()


    # Print the final report into a .txt file

DateTime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

with open("FinalReport.txt", "w") as RepoTXT:

    RepoTXT.write('\n\tFinal report file - Script created by Luca Rossini luca.rossini@unitus.it\n\n')
    RepoTXT.write('\t' + DateTime + "\n\n")
    RepoTXT.write('\n\tLeast Squares (LSF) step FitResults:\n\n')
    RepoTXT.write(fit_report(BestValues_DataFrame['FitResults'][0]))
    RepoTXT.write('\n\nBest Iteration name is: ' + BestIterationName + '\n')
    RepoTXT.write('\nTotal number of iterations in LSF step: ' + str(TotIter_LSF) + '\n')
    RepoTXT.write('\nTotal time of execution of the LSF step: ' + str(round(EndTime_LSF - StartTime_LSF, 3)) + ' seconds\n')
    RepoTXT.write('\nCheck the LeastSquares-BestValues.csv file for additional information about this first step\n')
    RepoTXT.write('\nTotal time of execution of the MCMC step: ' + str(round(EndTime_MCMC - EndTime_LSF, 3)) + ' seconds\n')


# Representative plots: LSF step just for comparison

plt.figure(1)
plt.plot(day_obs, yobs, label=f"Adult males")
plt.plot(day_obs, BestValues_DataFrame['FinalValues'][0], '--', label=f"Best LS fit")
plt.xlabel('Time (days)')
plt.ylabel('Population density')
plt.legend()

    # Briére Egg-Pupa

figure, axis_EP = plt.subplots(2, 2)

plt.suptitle('Briére Egg-Pupa')

axis_EP[0, 0].hist(BestValues_DataFrame['a_EP'][0:99], bins = 10, facecolor = 'g', alpha=0.75)
axis_EP[0, 0].set_xlabel('a values')
axis_EP[0, 0].set_ylabel('Frequency')

axis_EP[0, 1].hist(BestValues_DataFrame['T_L_EP'][0:99], bins = 10, facecolor = 'g', alpha=0.75)
axis_EP[0, 1].set_xlabel('T_L values')
axis_EP[0, 1].set_ylabel('Frequency')

axis_EP[1, 0].hist(BestValues_DataFrame['T_M_EP'][0:99], bins = 10, facecolor = 'g', alpha=0.75)
axis_EP[1, 0].set_xlabel('T_M values')
axis_EP[1, 0].set_ylabel('Frequency')

axis_EP[1, 1].hist(BestValues_DataFrame['m_EP'][0:99], bins = 10, facecolor = 'g', alpha=0.75)
axis_EP[1, 1].set_xlabel('m values')
axis_EP[1, 1].set_ylabel('Frequency')

    # Briére Pupa-Adult

figure, axis_PA = plt.subplots(2, 2)

plt.suptitle('Briére Pupa-Adult')

axis_PA[0, 0].hist(BestValues_DataFrame['a_PA'][0:99], bins = 10, facecolor = 'g', alpha=0.75)
axis_PA[0, 0].set_xlabel('a values')
axis_PA[0, 0].set_ylabel('Frequency')

axis_PA[0, 1].hist(BestValues_DataFrame['T_L_PA'][0:99], bins = 10, facecolor = 'g', alpha=0.75)
axis_PA[0, 1].set_xlabel('T_L values')
axis_PA[0, 1].set_ylabel('Frequency')

axis_PA[1, 0].hist(BestValues_DataFrame['T_M_PA'][0:99], bins = 10, facecolor = 'g', alpha=0.75)
axis_PA[1, 0].set_xlabel('T_M values')
axis_PA[1, 0].set_ylabel('Frequency')

axis_PA[1, 1].hist(BestValues_DataFrame['m_PA'][0:99], bins = 10, facecolor = 'g', alpha=0.75)
axis_PA[1, 1].set_xlabel('m values')
axis_PA[1, 1].set_ylabel('Frequency')

    # Briére Adult Survival

figure, axis_Sur = plt.subplots(2, 2)

plt.suptitle('Briére Adult survival')

axis_Sur[0, 0].hist(BestValues_DataFrame['a_Sur'][0:99], bins = 10, facecolor = 'g', alpha=0.75)
axis_Sur[0, 0].set_xlabel('a values')
axis_Sur[0, 0].set_ylabel('Frequency')

axis_Sur[0, 1].hist(BestValues_DataFrame['T_L_Sur'][0:99], bins = 10, facecolor = 'g', alpha=0.75)
axis_Sur[0, 1].set_xlabel('T_L values')
axis_Sur[0, 1].set_ylabel('Frequency')

axis_Sur[1, 0].hist(BestValues_DataFrame['T_M_Sur'][0:99], bins = 10, facecolor = 'g', alpha=0.75)
axis_Sur[1, 0].set_xlabel('T_M values')
axis_Sur[1, 0].set_ylabel('Frequency')

axis_Sur[1, 1].hist(BestValues_DataFrame['m_Sur'][0:99], bins = 10, facecolor = 'g', alpha=0.75)
axis_Sur[1, 1].set_xlabel('m values')
axis_Sur[1, 1].set_ylabel('Frequency')


plt.figure(5)

plt.hist(BestValues_DataFrame['ChiSquare'], bins=100, facecolor='g', alpha=0.75)
plt.title('Histogram of LFS-ChiSquare values')
plt.xlabel('ChiSquare value')
plt.ylabel('Frequence')



plt.show()
exit()






















