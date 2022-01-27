# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # The Multi-Color Prophet Problem notebook

# %% [markdown]
#  Contact: p.praat@student.uva.nl

# %%
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.stats import uniform
from numpy.random import default_rng
from collections import Counter
import matplotlib.pyplot as plt
from statistics import mean
import pandas as pd

from numpy import save
from numpy import load

from tqdm import tqdm # for the progress bar


# %% [markdown]
# _We next consider the following multi-color prophet problem. In this model n candidates arrive in uniform random order. Candidates are partitioned into k groups $C = {C_1,···,C_k}$. We write $n=(n1,...,nk)$ for the vector of groupsizes, i.e., $|C_j| = n_j$ ,for all 1 ≤ j ≤ k. We identify each of the groups with a distinct color and let c(i), vi denote the color and value of candidate i, respectively. The value vi that is revealed upon arrival of i, and is drawn independently from a given distribution Fi. We use $F = (F_1, . . . , Fn)$ to refer to the vector of distributions. We are also given a probability vector $p = (p_1, . . . , p_k)$. The goal is to select a candidate in an online manner in order to maximize the expectation of the value of the selected candidate, while selecting from each color with probability proportional to p. We distinguish between the basic setting in which $p_j$ is the proportion of candidates that belong to group j, i.e., $p_j = n_j/n$, and the general setting in which $p$ is arbitrary. We compare ourselves with the fair optimum, the optimal offline algorithm that respects the $p_j$ ’s._

# %% [markdown]
# ## The two algorithms presented in the paper

# %%
def middleBinomial(n):
    n_candidates = n
    n = 1000
    p = 0.5 #probability of coin succeeding
    choose = np.zeros((n+1,n+1)) # creating 'choose' variable -> number of combinations per number of successes
    for i in range(n+1):
        choose[i,0] = 1

    for i in range(1,n+1):
        for j in range(1,n+1):
            choose[i,j] = choose[i-1,j-1] + choose[i-1,j]

    n_combinations = choose[-1]
    probability = np.ones((n+1)) #probability of n successes
    r_probability = np.ones((n+1)) #reverse probability of n failures

    for i in range(1,n+1):
        probability[i] = probability[i - 1] * p
        r_probability[i] = r_probability[i - 1] * (1 - p);
        
        
    #max dist --> chance of getting at least certain amount of successes after all candidates
    x = n_combinations * probability * np.flip(r_probability) #calculating p of i successes in one try, by multiplying the p of this many successes, this many failures and all combinations in which they could have occurred
    x_cum = np.flip(np.cumsum(np.flip(x))) #calculating cumulative probability (p of getting at least i successes in one try)
    max_dist = 1 - pow(1-x_cum, n_candidates) #p of getting i successes after certain amount of tries

    #middle --> find highest number of successes where probability of reaching at least that is more than 0.5
    for i in np.arange(len(max_dist)-1, -1,-1):
        if max_dist[i] >= 0.5:
            #TODO: figure out why in the code they do divde this by n!
            middle = i #/ n ### question: binomial data comes in absolute n successes or fraction of total tries ??? ###
            break
        if i == 0.0:
            middle = 0
    return middle


# %%
def Finv(distribution, prob):
    lower, upper = 0.0,1.0
    if distribution == "uniform":
        return prob * (upper-lower)
    if distribution == "binomial":
        return scipy.stats.binom.ppf(prob, n=1000, p=0.5)
        

def Middle(distribution_type, n):
    if distribution_type == "uniform":
        rrange = 1.0
        return rrange * np.power(1.0 / 2, 1.0 / n)
    if distribution_type == "binomial":
        return middleBinomial(n)
        
        
def FairGeneralProphet (q, V, distribution_type):
    summ = 0.0
    for i in range(0,len(V)): #value < 1 reaches a drop!
        if V[i] >= Finv(distribution_type, (1.0 - (q[i] / (2.0 - summ)))):
#         if V[i] >= Finv(distribution_type, (1- (q[i]/2)/(1-(summ/2)))):
            return i
        summ += q[i]

def FairIIDProphet(Values, distribution_type):
    for i in range(0, len(Values)):
        p = (2.0 / 3.0) / len(Values)
        if Values[i] >= Finv(distribution_type, (1.0 - p / (1.0 - p * i))):
            return i



# %%
# Implemented according to the function “ComputeSolutionOneHalf” in unfair-prophet.cc
def SC_algorithm(Values, distribution_type):
    middleValue = Middle(distribution_type, len(Values))
    for i in range(0, len(Values)):
        if Values[i] >= middleValue:
            return i


# Implemented according to the function “ComputeSolutionOneMinusOneE” in unfair-prophet.cc
def EHKS_algorithm(Values, distribution_type):
    for i in range(0, len(Values)):
        if Values[i] >= Finv(distribution_type, (1.0 - (1.0 / len(Values)))):
            return i

# Implemented according to the function “ComputeSolutionDiffEq” in unfair-prophet.cc
def DP_algorithm(Values, distribution_type):
#     TODO: still need to figure out if these precomputed thresholds hold!

    diff_solution_50 = np.loadtxt("diff_solution_50.txt", delimiter=", ")
    diff_solution_1000 = np.loadtxt("diff_solution_1000.txt", delimiter=", ")

    if len(Values) == 50:
        diff_solution = diff_solution_50
    else:
        diff_solution = diff_solution_1000
        
    for i in range(0, len(Values)):
        if Values[i] >= Finv(distribution_type, np.power(diff_solution[i], (1.0 / (len(Values) - 1)))):
            return i


# %% [markdown]
# ## The experiments

# %% [markdown]
# _"We focus on the case, where values are distributed i.i.d. and each candidate is a group on its own. We consider two settings. In the first one the input stream consists of 50 samples from the uniform distribution in range [0, 1], and in the second one the input consists of 1000 samples from the binomial distribution with 1000 trials and 1/2 probability of success of a single trial. For better comparability with existing algorithms, in both cases we assume each candidate is a group on its own. We run each algorithm 50, 000 times."_

# %% [markdown]
# ### Uniform Distribution

# %% [markdown]
# _Here, as mentioned in the paper, the "input stream consists of 50 samples from the uniform distribution in range [0, 1]"_

# %%
"""
:param distribution_type: either "uniform" or "binomial"
:param size: number of candidates
:returns q: 
:returns V:
"""
def generateDistribution(distribution_type, size):
    rng = default_rng()
    n = size
    if distribution_type == "uniform":
        q, V = [1/n] * n , rng.uniform(low=0.0, high=1.0, size=n)
    elif distribution_type == "binomial":
        q, V = [1/n] * n , rng.binomial(n=1000, p=.5, size=n)
    return q,V

"""
:param algorithm: string either "FairGeneralProphet", "FairIIDProphet", "SC", "EHKS" or, "DP"
:param N_experimentReps: the number of times the algorithm needs to run
:param distribution_type: either "uniform" or "binomial"
:param n_candidates: interger with the number of candidates in each experiment
:returns arrivalPositionsChosen: array containing which candidate position was chosen
:returns chosenValues: array contraining the values of each picked/selected candidate
"""
def runExperiment(algorithm, N_experimentReps, distribution_type, n_candidates):
    arrivalPositionsChosen, chosenValues = [0]*n_candidates, []
    for _ in tqdm(range(0, N_experimentReps)):
        q, Values = generateDistribution(distribution_type, n_candidates)
        
        if algorithm == "FairGeneralProphet":
                result = FairGeneralProphet(q, Values, distribution_type)
        elif algorithm == "FairIIDProphet":
                result = FairIIDProphet(Values, distribution_type)
        elif algorithm == "SC":
                result = SC_algorithm(Values, distribution_type)
        elif algorithm =="EHKS":
                result = EHKS_algorithm(Values, distribution_type)
        elif algorithm == "DP":
                result = DP_algorithm(Values, distribution_type)
                
                
        if result != None:
            arrivalPositionsChosen[result] += 1
            chosenValues.append(Values[result])
            
        if result == None: chosenValues.append(0)
    return arrivalPositionsChosen, chosenValues

# %%
# #Plotting the results for 50k experiments

# arrivalPositionsChosenFairPA, a = runExperiment(algorithm="FairGeneralProphet", N_experimentReps=50000, 
#                                                 distribution_type="uniform", n_candidates=50)
    
# arrivalPositionsChosenFairIID, b = runExperiment(algorithm="FairIIDProphet", N_experimentReps=50000, 
#                                                 distribution_type="uniform", n_candidates=50)
    
# arrivalPositionsChosenSC, c = runExperiment(algorithm="SC", N_experimentReps=50000, 
#                                                 distribution_type="uniform", n_candidates=50)
    
# arrivalPositionsChosenEHKS, d = runExperiment(algorithm="EHKS", N_experimentReps=50000, 
#                                                 distribution_type="uniform", n_candidates=50)
# arrivalPositionsChosenDP, e = runExperiment(algorithm="DP", N_experimentReps=50000, 
#                                                 distribution_type="uniform", n_candidates=50)

# plt.plot(range(0,50), arrivalPositionsChosenFairPA, label="Fair PA")
# plt.plot(range(0,50), arrivalPositionsChosenFairIID, label="Fair IID")
# plt.plot(range(0,50), arrivalPositionsChosenEHKS, label="EHKS")
# plt.plot(range(0,50), arrivalPositionsChosenSC, label="SC")
# plt.plot(range(0,50), arrivalPositionsChosenDP, label="DP")
# plt.plot(range(0,50), range(0,4000,80), label="replicate CFHOV for scale")
# plt.title("50k experiments, discarding None results")
# plt.xlabel("Arrival position")
# plt.ylabel("Num Picked")
# plt.legend(loc="upper right")
# plt.savefig("images/50kExperiments.png")

# %% [markdown]
# This led us to examining two approaches. One in which we increase the number of experiments to 100k, and one where we run 50k experiments and keep repeating each experiment untill we don't get a None. The first one has been done above, and seems to confirm the found results.
#
# It is also worth noting that just skipping over None results does not lead to the same results, only increasing the number of experiments. As shown here

# %%
# #Plotting the results for 100k experiments

# arrivalPositionsChosenFairPA, a = runExperiment(algorithm="FairGeneralProphet", N_experimentReps=50000*2, 
#                                                 distribution_type="uniform", n_candidates=50)
    
# arrivalPositionsChosenFairIID, b = runExperiment(algorithm="FairIIDProphet", N_experimentReps=50000*2, 
#                                                 distribution_type="uniform", n_candidates=50)
    
# arrivalPositionsChosenSC, c = runExperiment(algorithm="SC", N_experimentReps=50000*2, 
#                                                 distribution_type="uniform", n_candidates=50)
    
# arrivalPositionsChosenEHKS, d = runExperiment(algorithm="EHKS", N_experimentReps=50000*2, 
#                                                 distribution_type="uniform", n_candidates=50)

# arrivalPositionsChosenDP, e = runExperiment(algorithm="DP", N_experimentReps=50000*2, 
#                                                 distribution_type="uniform", n_candidates=50)

# %% [markdown]
# ### Average values of chosen candidates

# %%
# print("The average value of the chosen candidate in the uniform distribution: \n")
# print("FairPA: ", mean(a), "(should be 0.501)")
# print("FairIID: ", mean(b), "(should be 0.661)")
# print("SK: ", mean(c), "(should be 0.499)")
# print("EHKS: ", mean(d), "(should be 0.631)")
# print("SP: ", mean(e), "(should be 0.751)")

# %% [markdown]
#
# **Statement 1:** _In conclusion, for both settings, both our algorithms Algorithm 2 and Algorithm 3 provide perfect fairness, while giving 66.71% and 88.01% (for the uniform case), and 58.12% and 75.82% (for the binomial case), of the value of the optimal, but unfair, online algorithm._
#
# Here the unfair online algorihtm must either SC or EHKS, something else is not mentioned about online algorithms. Also in the codebase, the unfair algorithms refer to these. Thus we will take a look at both, and see if these results are replicable.
#

# %%
# print("Uniform case, for FairPA")
# print("Assuming DP as the 'optimal, but unfair, online algorithm' :", mean(a) / mean(e) *100, "%")

# print("\n Uniform case, for FairIID")
# print("Assuming DP as the 'optimal, but unfair, online algorithm' :", mean(b) / mean(e) *100, "%")



# %% [markdown]
# Both of these approaches thus do not (nearly) approach the results as presented in the paper, and seem to be verifiable in Figure 2 in the paper. Thus, the question is raised which algorithm is used to calculate the percentages mentioned in **Statement 1**.

# %% [markdown]
# ## Binomial distribution

# %%
# arrivalPositionsChosenFairPA, FairPA_values = runExperiment(algorithm="FairGeneralProphet", N_experimentReps=50000, 
#                                                 distribution_type="binomial", n_candidates=1000)
# save('data/FairPA_positions.npy', arrivalPositionsChosenFairPA)
# save('data/FairPA_values.npy', FairPA_values)

# %%
# arrivalPositionsChosenFairIID, FairIID_values = runExperiment(algorithm="FairIIDProphet", N_experimentReps=50000, 
#                                                 distribution_type="binomial", n_candidates=1000)
# save('data/FairIID_positions.npy', arrivalPositionsChosenFairIID)
# save('data/FairIID_values.npy', FairIID_values)

# %%
arrivalPositionsChosenSC, SC_values = runExperiment(algorithm="SC", N_experimentReps=50000, 
                                               distribution_type="binomial", n_candidates=1000)

save('data/SC_positions.npy', arrivalPositionsChosenSC)
save('data/SC_values.npy', SC_values)

arrivalPositionsChosenSC

# %%
# arrivalPositionsChosenEHKS, EHKS_values = runExperiment(algorithm="EHKS", N_experimentReps=50000, 
#                                                 distribution_type="binomial", n_candidates=1000)

# save('data/EHKS_positions.npy', arrivalPositionsChosenEHKS)
# save('data/EHKS_values.npy', EHKS_values)

# %%
# arrivalPositionsChosenDP, DP_values = runExperiment(algorithm="DP", N_experimentReps=50000, 
#                                                 distribution_type="binomial", n_candidates=1000)

# save('data/DP_positions.npy', arrivalPositionsChosenDP)
# save('data/DP_values.npy', DP_values)

# %%
plt.plot(range(0,1000), load('data/FairPA_positions.npy'), label="FairPA")
plt.plot(range(0,1000), load('data/FairIID_positions.npy'), label="Fair IID")
plt.plot(range(0,1000), load('data/EHKS_positions.npy'), label="EHKS")
plt.plot(range(0,1000), load('data/SC_positions.npy')*25, label="SC")
plt.plot(range(0,1000), load('data/DP_positions.npy'), label="DP")
plt.xlabel("Arrival position")
plt.ylabel("Num Picked")
plt.title("Binomial distribution with 1k candidates, and 50k experiments")
plt.legend(loc="upper right")
plt.savefig("binomial.png")

# %%
arrivalPositionsChosenFairPA, FairPA_values = runExperiment(algorithm="FairGeneralProphet", N_experimentReps=50000*2, 
                                                distribution_type="binomial", n_candidates=1000)
save('data/FairPA_positions100k.npy', arrivalPositionsChosenFairPA)
save('data/FairPA_values100k.npy', FairPA_values)

arrivalPositionsChosenFairIID, FairIID_values = runExperiment(algorithm="FairIIDProphet", N_experimentReps=50000*2, 
                                                distribution_type="binomial", n_candidates=1000)
save('data/FairIID_positions100k.npy', arrivalPositionsChosenFairIID)
save('data/FairIID_values100k.npy', FairIID_values)

arrivalPositionsChosenSC, SC_values = runExperiment(algorithm="SC", N_experimentReps=50000*2, 
                                               distribution_type="binomial", n_candidates=1000)

save('data/SC_positions100k.npy', arrivalPositionsChosenSC)
save('data/SC_values100k.npy', SC_values)

arrivalPositionsChosenEHKS, EHKS_values = runExperiment(algorithm="EHKS", N_experimentReps=50000*2, 
                                                distribution_type="binomial", n_candidates=1000)

save('data/EHKS_positions100k.npy', arrivalPositionsChosenEHKS)
save('data/EHKS_values100k.npy', EHKS_values)

arrivalPositionsChosenDP, DP_values = runExperiment(algorithm="DP", N_experimentReps=50000*2, 
                                                distribution_type="binomial", n_candidates=1000)

save('data/DP_positions100k.npy', arrivalPositionsChosenDP)
save('data/DP_values100k.npy', DP_values)


# %% [markdown]
# ## Extension

# %% [markdown]
# Don't need mathmathical underpinning! Just mention that it is possible/relevant.

# %%
import pandas as pd

def FairGeneralProphetExtended(q, V, distribution_type, parameter_value):
    s = 0.0
    for i in range(0,len(V)): #value < 1 reaches a drop!
        
        assert (1.0 - (q[i] / (2 - s))) == (1- (q[i]/2)/(1-(s/2)))
        
        if V[i] >= Finv(distribution_type, (1.0 - (q[i] / (parameter_value - s)))):
#         if V[i] >= Finv(distribution_type, (1- (q[i]/2)/(1-(summ/2)))):
            return i
        s += q[i]

def FairIIDProphetExtended(Values, distribution_type, parameter_value):
    for i in range(0, len(Values)):
        p = (2.0 / 3.0) / len(Values)
        
        ii = int(i)
        n = len(Values)
        
#         print((1.0 - p / (1.0 - p * ii)))
#         print((2 / 3*n) / (1 - 2*(ii-1)/3*n))
        
#         assert (1.0 - p / (1.0 - p * ii)) == (2/3*n)/(1-2(ii-1)/3*n)
        
        p = ((parameter_value) / 3.0) / len(Values)
        if Values[i] >= Finv(distribution_type, (1.0 - p / (1.0 - p * i))):
            return i
        
def runExperimentExtended(algorithm, N_experimentReps, distribution_type, n_candidates, parameter_value):
    arrivalPositionsChosen, chosenValues, chosenValuesExcludeNone = [0]*n_candidates, [], []
    nones = 0
    for _ in tqdm(range(0, N_experimentReps)):
        q, Values = generateDistribution(distribution_type, n_candidates)
        
        if algorithm == "FairGeneralProphet":
                result = FairGeneralProphetExtended(q, Values, distribution_type, parameter_value)
        elif algorithm == "FairIIDProphet":
                result = FairIIDProphetExtended(Values, distribution_type, parameter_value)
        elif algorithm == "SC":
                result = SC_algorithm(Values, distribution_type)
        elif algorithm =="EHKS":
                result = EHKS_algorithm(Values, distribution_type)
        elif algorithm == "DP":
                result = DP_algorithm(Values, distribution_type)
        if result != None:
            arrivalPositionsChosen[result] += 1
            chosenValues.append(Values[result])
            chosenValuesExcludeNone.append(Values[result])
            
        if result == None: 
            chosenValues.append(0)
            nones += 1
            
        
    noneRate = nones/N_experimentReps
        
    return noneRate, sum(chosenValues)/N_experimentReps, sum(chosenValuesExcludeNone)/N_experimentReps, arrivalPositionsChosen, mean(chosenValues), mean(chosenValuesExcludeNone)

df = pd.DataFrame(columns=['Parameter value', 'None rate', "Mean value (None=0)", "Mean value (excluding None)"])
for param in [1.0, 1.25,1.5,1.75,2.0, 2.25, 2.5, 2.75, 3.0, 3.25 ,3.5]:
    nonerate, avg_include, avg_exclude, chosen_positions, avg_include_own, avg_exclude_own = runExperimentExtended(algorithm="FairIIDProphet", N_experimentReps=50000, 
                                                distribution_type="binomial", n_candidates=100, parameter_value=param)
    print("Nonerate: ", nonerate * 100, "%")
#     print("Average value of the chosen candidate with none’s as 0 value (in whole group): ", avg_include)
#     print("Average value of the chosen candidate with None's excluded (in whole group): ", avg_exclude)
    
    print("Average value of the chosen candidate with none’s as 0 value (in own group): ", avg_include_own)
    print("Average value of the chosen candidate with None's excluded (in own group): ", avg_exclude_own)
    
    a_series = pd.Series([param,nonerate,avg_include_own,avg_exclude_own], index = df.columns)
    df = df.append(a_series, ignore_index=True)
    
#     df = df.append([[param,avg_include,avg_exclude]], ignore_index=True)

    plt.plot(range(0,50), chosen_positions, label= str("γ = " + str(param)))
# plt.plot(range(0,50), range(0,4000,80), label="replicate CFHOV for scale")
plt.xlabel("Arrival position")
plt.ylabel("Num Picked")
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4)
plt.save()

# %%
df