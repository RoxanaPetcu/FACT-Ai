import numpy as np
import pandas as pd
import typing


class SecretaryInstance:

    def __init__(self, score, color, p):
        """The SecretaryInstance class creates a candidate object

        Args:
            score (float): The score of the candidate
            color (string): The group of the candidate
            p (float): The group probability of being selected
        """
        self.score = score
        self.color = color
        self.p = p
        self.ismax = False

def get_synthetic_data(*args) -> typing.List[SecretaryInstance]:
    """This method receives necessary arguments (list of colors, probabilities and size of groups)
    and returns a list of candidates

    Returns:
        [SecretaryInstance]: List of candidates
    """
    
    np.random.seed(0)
    
    colors, probabilities, n = args[0], args[1], args[2]   
    synthetic_data = []

    for i in range(len(n)):

        new_group = np.random.uniform(0, 1, n[i])
        for item in new_group:
            synthetic_data.append(SecretaryInstance(item, colors[i], probabilities[i]))

    return synthetic_data

def get_bank_data(path, colors, probabilities) -> typing.Tuple[typing.List[SecretaryInstance], typing.List[int]]:
    """This method constructs the bank data.

    Args:
        path (string): Directory for reading data and writing results
        colors ([string]): List of colors
        probabilities ([float]): List of group probabilities

    Returns:
        [SecretaryInstance]: List of candidates
        [int]: Size of groups
    """
    
    bank_data = pd.read_csv(path, sep = ';')
    call_duration = bank_data[['age', 'duration']]
    synthetic_data = []
    n = []
    
    for i in range(len(call_duration)):
        
        if call_duration.loc[i].age <= 30:
            individual = SecretaryInstance(call_duration.loc[i].duration, colors[0], probabilities[0])
        elif call_duration.loc[i].age > 30 and call_duration.loc[i].age <= 40:
            individual = SecretaryInstance(call_duration.loc[i].duration, colors[1], probabilities[1])
        elif call_duration.loc[i].age > 40 and call_duration.loc[i].age <= 50:
            individual = SecretaryInstance(call_duration.loc[i].duration, colors[2], probabilities[2])
        elif call_duration.loc[i].age > 50 and call_duration.loc[i].age <= 60:
            individual = SecretaryInstance(call_duration.loc[i].duration, colors[3], probabilities[3])
        elif call_duration.loc[i].age > 60:
            individual = SecretaryInstance(call_duration.loc[i].duration, colors[4], probabilities[4])
            
        synthetic_data.append(individual)
    
    for color in colors:
        n.append(len([item for item in synthetic_data if item.color == color]))
    
    return synthetic_data, n

def get_pokec_measurements(pokec_data_split) -> typing.Tuple[typing.List[int], typing.List[int], typing.List[int]]:
    """This method receives unprocessed data and extracts the needed features

    Args:
        pokec_data_split ([string]): Unprocessed data in the form of list of sentences

    Returns:
        [int]: The height of the user
        [int]: The weight of the user
        [int]: The ID of the user
    """
    
    height, weight, user_id = [], [], []

    txt = pokec_data_split[0][8]
    body_measurements = [int(s) for s in txt.split() if s.isdigit()]

    height.append(body_measurements[0])
    weight.append(body_measurements[1])
    user_id.append(int(pokec_data_split[0][0]))

    for i in range(1, len(pokec_data_split)-1):

        try:
            txt = pokec_data_split[i][9]
            body_measurements = [int(s) for s in txt.split() if s.isdigit()]
        except ValueError:
            print("Value Error")
            
        if (len(body_measurements) == 2 and body_measurements[0] >= 100 and body_measurements[0] < 230 and body_measurements[1] > 25):
            height.append(body_measurements[0])
            weight.append(body_measurements[1])
            user_id.append(int(pokec_data_split[i][1]))
            
    return height, weight, user_id

def get_pokec_data(path, path_relationships, colors, probabilities) -> typing.Tuple[typing.List[SecretaryInstance], typing.List[int]]:
    """This function constructs the pokec data

    Args:
        path (string): Directory for reading the dataset containing information on the app users
        path_relationships (string): Directory for reading the number of followers on the app users
        colors ([string]): List of colors
        probabilities ([float]): List of group probabilities

    Returns:
        [SecretaryInstance]: List of candidates
        [int]: Size of groups
    """

    synthetic_data, n = [], []

    with open(path) as f:
        pokec_data = f.readlines()
        
    pokec_data_split = [l.split('\t') for l in '\t'.join(pokec_data).split('\n')]
    height, weight, user_id = get_pokec_measurements(pokec_data_split)
    
    with open(path_relationships) as f:
        pokec_data_relationships = f.readlines()
        
    followers = [0] * 1700000

    for item in pokec_data_relationships:
        followers[int(item.split()[0])] += 1
        followers[int(item.split()[1])] += 1
    
    bmi = np.multiply(np.divide(weight, np.power(height, 2)), 10000)

    bmi_sorted, user_followers_sorted = zip(*sorted(zip(bmi, np.array(followers)[user_id])))
    bmi_sorted = list(bmi_sorted)
    
    under = len([item for item in bmi_sorted if item < 18.5])
    normal = len([item for item in bmi_sorted if item < 25])
    over = len([item for item in bmi_sorted if item < 30])
    obese1 = len([item for item in bmi_sorted if item < 35])
    
    for i in range(len(user_followers_sorted)):
        
        if i < under:
            individual = SecretaryInstance(user_followers_sorted[i], colors[0], probabilities[0])
        elif i >= under and i < normal:
            individual = SecretaryInstance(user_followers_sorted[i], colors[1], probabilities[1])
        elif i >= normal and i < over:
            individual = SecretaryInstance(user_followers_sorted[i], colors[2], probabilities[2])
        elif i >= over and i < obese1:
            individual = SecretaryInstance(user_followers_sorted[i], colors[3], probabilities[3])
        elif i >= obese1:
            individual = SecretaryInstance(user_followers_sorted[i], colors[4], probabilities[4])
            
        synthetic_data.append(individual)
    
    for color in colors:
        n.append(len([item for item in synthetic_data if item.color == color]))

    return synthetic_data, n
