
# List of the parameters in input into the scripts 'ODEdroso.py' and 'MCMC-ODE-Briere.py'.

import multiprocessing as mp

# Parameters for the least squares fitting
# Note: MaxEstimations_LeastSq should be multiple of 4

MaxIterations_LeastSq = mp.cpu_count() * 2
MaxEstimations_LeastSq = mp.cpu_count() * 2
MaxIterations_GeneticAlgorithm = 2

# Parameters for the MCMC algorithm (it uses the Ray cluster!)

NumberChains_MCMC = 384
NumberIterations_PerChain_MCMC = 20000

# Initial conditions for the LeastSquares-MCMC
# These initial conditions are determinant for the whole process

    # Egg-Pupa stages

a_MCMC_EP = 3.30 * pow(10, -4)
T_L_MCMC_EP = 1.0
T_M_MCMC_EP = 29.0
m_MCMC_EP = 2.9

    # Pupa-Adult stage

a_MCMC_PA = 1.20 * pow(10, -4)
T_L_MCMC_PA = 5.0
T_M_MCMC_PA = 31.0
m_MCMC_PA = 4.0

    # Adult survival

a_MCMC_Sur = 2.8 * pow(10, -4)
T_L_MCMC_Sur = 3.0
T_M_MCMC_Sur = 30.0
m_MCMC_Sur = 3.7


# Theoretical parameters

    # Briere: Egg-Pupa stages

a_EP = 1.5909 * pow(10, -4)
T_L_EP = 2.0919
T_M_EP = 32.088
m_EP = 4.0

    # Briere: Pupa-Adult stage

a_PA = 2.3699 * pow(10, -4)
T_L_PA = 4.0
T_M_PA = 33.164
m_PA = 4.0

    # Briere: Adult survival

a_Sur = 6.8417 * pow(10, -5)
T_L_Sur = -3.0
T_M_Sur = 30.034
m_Sur = 2.50

    # Fertility

alpha = 659.06
gamma = 88.53
Lambda = 52.32
delta = 6.06
tau = 22.87
T_low = -3
T_max = 39

    # Mortality

k_mort = 1.0
T_MAX_mort = 23.4265085 
rho_mort = -5.5455493

    # Sex ratio

SR = 0.5

    # Initial conditions for the ODE system

E_0 = 50
L1_0 = 0
L2_0 = 0
L3_0 = 0
P_0 = 0
Am_0 = 0
Anmf_0 = 0
Amf_0 = 97


# Plotting results

    # Briere best fit parameters from LSF step

        # Egg-Pupa

a_EP_BestLSF = 1.5909 * pow(10, -4)
T_L_EP_BestLSF = 2.0919
T_M_EP_BestLSF = 32.088
m_EP_BestLSF = 4.0

        # Pupa-Adult

a_PA_BestLSF = 2.3699 * pow(10, -4)
T_L_PA_BestLSF = 4.0
T_M_PA_BestLSF = 33.164
m_PA_BestLSF = 4.0

        # Adult survival

a_Sur_BestLSF = 6.8417 * pow(10, -5)
T_L_Sur_BestLSF = -3.00
T_M_Sur_BestLSF = 30.034
m_Sur_BestLSF = 2.50

    # Briere best fit parameters from MCMC step

        # Egg-Pupa

a_EP_BestMCMC = 1.5909 * pow(10, -4)
T_L_EP_BestMCMC = 2.0919
T_M_EP_BestMCMC = 32.088
m_EP_BestMCMC = 4.0

        # Pupa-Adult

a_PA_BestMCMC = 2.3699 * pow(10, -4)
T_L_PA_BestMCMC = 4.0
T_M_PA_BestMCMC = 33.164
m_PA_BestMCMC = 4.0

        # Adult survival

a_Sur_BestMCMC = 6.8417 * pow(10, -5)
T_L_Sur_BestMCMC = -3.00
T_M_Sur_BestMCMC = 30.034
m_Sur_BestMCMC = 2.50




