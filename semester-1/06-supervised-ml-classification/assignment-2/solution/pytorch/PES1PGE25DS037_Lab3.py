# PES1PGE25DS037_Lab3.py
# ID3 Decision Tree building blocks - PyTorch implementation
import torch


def get_entropy_of_dataset(tensor: torch.Tensor):
    """
    Calculate the entropy of the entire dataset.
    Formula: Entropy = -Sum(p_i * log2(p_i)) where p_i is the probability of class i

    Args:
        tensor (torch.Tensor): Input dataset as a tensor, where the last column is the target.

    Returns:
        float: Entropy of the dataset.
    """
    target = tensor[:, -1]
    _, counts = torch.unique(target, return_counts=True)
    probabilities = counts.float() / counts.sum()
    entropy = -torch.sum(probabilities * torch.log2(probabilities))
    return float(entropy.item())


def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int):
    """
    Calculate the average information (weighted entropy) of an attribute.
    Formula: Avg_Info = Sum((|S_v|/|S|) * Entropy(S_v)) where S_v is subset with attribute value v.

    Args:
        tensor (torch.Tensor): Input dataset as a tensor.
        attribute (int): Index of the attribute column.

    Returns:
        float: Average information of the attribute.
    """
    total_samples = tensor.shape[0]
    values, counts = torch.unique(tensor[:, attribute], return_counts=True)

    avg_info = 0.0
    for value, count in zip(values, counts):
        subset = tensor[tensor[:, attribute] == value]
        weight = count.item() / total_samples
        avg_info += weight * get_entropy_of_dataset(subset)

    return float(avg_info)


def get_information_gain(tensor: torch.Tensor, attribute: int):
    """
    Calculate Information Gain for an attribute.
    Formula: Information_Gain = Entropy(S) - Avg_Info(attribute)

    Args:
        tensor (torch.Tensor): Input dataset as a tensor.
        attribute (int): Index of the attribute column.

    Returns:
        float: Information gain for the attribute (rounded to 4 decimals).
    """
    gain = get_entropy_of_dataset(tensor) - get_avg_info_of_attribute(tensor, attribute)
    return round(float(gain), 4)


def get_selected_attribute(tensor: torch.Tensor):
    """
    Select the best attribute based on highest information gain.

    Returns a tuple with:
    1. Dictionary mapping attribute indices to their information gains
    2. Index of the attribute with highest information gain

    Example: ({0: 0.123, 1: 0.768, 2: 1.23}, 2)

    Args:
        tensor (torch.Tensor): Input dataset as a tensor.

    Returns:
        tuple: (dict of attribute:index -> information gain, index of best attribute)
    """
    num_attributes = tensor.shape[1] - 1
    information_gains = {
        attribute: get_information_gain(tensor, attribute)
        for attribute in range(num_attributes)
    }
    selected_attribute = max(information_gains, key=information_gains.get)
    return information_gains, selected_attribute
