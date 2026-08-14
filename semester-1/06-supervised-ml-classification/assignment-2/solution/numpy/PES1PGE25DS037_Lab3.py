# PES1PGE25DS037_Lab3.py
# ID3 Decision Tree building blocks - NumPy implementation
import numpy as np


def get_entropy_of_dataset(data: np.ndarray) -> float:
    """
    Calculate the entropy of the entire dataset using the target variable (last column).

    Args:
        data (np.ndarray): Dataset where the last column is the target variable

    Returns:
        float: Entropy value calculated using the formula:
               Entropy = -Sum(p_i * log2(p_i)) where p_i is the probability of class i
    """
    target = data[:, -1]
    _, counts = np.unique(target, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy)


def get_avg_info_of_attribute(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the average information (weighted entropy) of a specific attribute.

    Args:
        data (np.ndarray): Dataset where the last column is the target variable
        attribute (int): Index of the attribute column to calculate average information for

    Returns:
        float: Average information calculated using the formula:
               Avg_Info = Sum((|S_v|/|S|) * Entropy(S_v))
               where S_v is subset of data with attribute value v
    """
    total_samples = len(data)
    values, counts = np.unique(data[:, attribute], return_counts=True)

    avg_info = 0.0
    for value, count in zip(values, counts):
        subset = data[data[:, attribute] == value]
        weight = count / total_samples
        avg_info += weight * get_entropy_of_dataset(subset)

    return float(avg_info)


def get_information_gain(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the Information Gain for a specific attribute.

    Args:
        data (np.ndarray): Dataset where the last column is the target variable
        attribute (int): Index of the attribute column to calculate information gain for

    Returns:
        float: Information gain calculated using the formula:
               Information_Gain = Entropy(S) - Avg_Info(attribute)
               Rounded to 4 decimal places
    """
    gain = get_entropy_of_dataset(data) - get_avg_info_of_attribute(data, attribute)
    return round(float(gain), 4)


def get_selected_attribute(data: np.ndarray) -> tuple:
    """
    Select the best attribute based on highest information gain.

    Args:
        data (np.ndarray): Dataset where the last column is the target variable

    Returns:
        tuple: A tuple containing:
            - dict: Dictionary mapping attribute indices to their information gains
            - int: Index of the attribute with the highest information gain
    """
    num_attributes = data.shape[1] - 1
    information_gains = {
        attribute: get_information_gain(data, attribute)
        for attribute in range(num_attributes)
    }
    selected_attribute = max(information_gains, key=information_gains.get)
    return information_gains, selected_attribute
