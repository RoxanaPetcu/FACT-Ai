import random
import pickle
import typing
import numpy as np
from data import get_synthetic_data, get_bank_data, get_pokec_data
from secretary import secretary_algorithm, one_color_secretary_algorithm, multiple_color_secretary_algorithm, multiple_color_thresholds


def secretary_experiment(candidates, *args) -> typing.Tuple[typing.Tuple, typing.Tuple, typing.Tuple]:
    """Runs three secretary algorithms on the given data: Two baselines, namely the Secretary Algorithm and the
    Single Color Secretary Algorithm; One fair optim algorithm names Multiple Color Secretary Algorithm

    Args:
        candidates ([SecretaryInstance]): List of SecretaryInstance objects; each instance contains the candidate's score, color and group probability
        args (tuple): Necessary arguments, namely the list of colors, probabilities and size of groups

    Returns:
        tuple: Chosen candidates by the Secretary Algorithm
        tuple: Chosen candidates by the Single Color Secretary Algorithm
        tuple: Chosen candidates by the Multiple Color Secretary Algorithm
    """

    colors, probabilities, n = args[0], args[1], args[2]
    thresholds = multiple_color_thresholds(probabilities)
    n_iterations = 20000
    
    for i in range(len(thresholds)):
        thresholds[i] = int(thresholds[i] * sum(n))

    results_SA, results_SCSA, results_MCSA = [], [], []

    max_colors = {color: None for color in colors}
    for color in colors:
        max_colors[color] = np.max([x.score for x in candidates if x.color == color])

    for i in range(n_iterations):
        
        if i % 1000 == 0:
            print(i)

        random.shuffle(candidates)
        result_SA = secretary_algorithm(candidates, max_colors)
        results_SA.append(result_SA)
        
        result_SCSA = one_color_secretary_algorithm(candidates, max_colors, colors, probabilities)
        results_SCSA.append(result_SCSA)
    
        result_MCSA = multiple_color_secretary_algorithm(candidates, max_colors, colors, thresholds)
        results_MCSA.append(result_MCSA)

    return results_SA, results_SCSA, results_MCSA

def synthetic_experiment() -> typing.Tuple:
    """Sets parameters and runs the synthetic data experiment"""
    
    colors = ['red', 'green', 'blue', 'yellow']
    probabilities = [0.25, 0.25, 0.25, 0.25]
    n = [10, 100, 1000, 10000]
    
    synthetic_data = get_synthetic_data(colors, probabilities, n)
    results = secretary_experiment(synthetic_data, colors, probabilities, n)

    return results

def unbalanced_synthetic_experiment() -> typing.Tuple:
    """Sets parameters and runs the unbalanced synthetic data experiment"""

    colors = ['red', 'green', 'blue', 'yellow']
    probabilities = [0.3, 0.25, 0.25, 0.2]
    n = [10, 100, 1000, 10000]

    synthetic_data = get_synthetic_data(colors, probabilities, n)
    results = secretary_experiment(synthetic_data, colors, probabilities, n)
    
    return results
    
def bank_experiment(path) -> typing.Tuple[typing.Tuple, typing.List[int]]:
    """Sets parameters and runs the feedback maximization experiment

    Args:
        path (string): Directory for reading data and writing results
    """
    
    colors = ["under 30", "31-40", "41-50", "51-60", "over 60"]
    probabilities = [0.2, 0.2, 0.2, 0.2, 0.2]
    n = []

    bank_data, n = get_bank_data(path, colors, probabilities)
        
    results = secretary_experiment(bank_data, colors, probabilities, n)
    
    return results, n

def pokec_experiment(path_profiles, path_followers) -> typing.Tuple[typing.Tuple, typing.List[int]]:
    """Sets parameters and runs the influence maximization experiment

    Args:
        path_profiles (string): Directory for reading the profiles data
        path_followers (string): Directory for reading the followers data
    """

    colors = ["Under", "Normal", "Over", "Obese1", "Obese2"]
    
    probabilities = [0.2, 0.2, 0.2, 0.2, 0.2]
    n = []

    # pokec_data, n = get_pokec_data(path_profiles, path_followers, colors, probabilities)

    with open('data/pokec_dataset.pickle', 'rb') as f:
        pokec_data = pickle.load(f)

    for i in range(len(colors)):
        n.append(len([item.color for item in pokec_data if item.color == colors[i]]))

    results = secretary_experiment(pokec_data, colors, probabilities, n)
    
    return results, n

def run_experiments():
    """Runs all four experiments for the Secretary Problem

    Args:
        path (string): Directory for reading data and writing results
    """
    
    results = synthetic_experiment()
    results2 = unbalanced_synthetic_experiment()
    # results3, n_bank = bank_experiment('data/bank_raw.csv')
    # results4, n_pokec = pokec_experiment('data/soc-pokec-profiles.txt', 'data/soc-pokec-relationships.txt')
    
    with open('results/results_synthetic1.pickle', 'wb') as f:
        pickle.dump(results, f)
        
    with open('results/results_synthetic2.pickle', 'wb') as f:
        pickle.dump(results2, f)
        
    # with open('results/results_bank.pickle', 'wb') as f:
    #     pickle.dump(results3, f)
        
    # with open('results/results_bank_args.pickle', 'wb') as f:
    #     pickle.dump(n_bank, f)
        
    # with open('results/results_pokec.pickle', 'wb') as f:
    #     pickle.dump(results4, f)
        
    # with open('results/results_pokec_args.pickle', 'wb') as f:
    #     pickle.dump(n_pokec, f)
