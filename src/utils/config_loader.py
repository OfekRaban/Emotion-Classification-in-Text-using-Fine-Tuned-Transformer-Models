import yaml
import torch


def load_class_weights(config_path: str, device: torch.device) -> torch.Tensor:
    """
    Load class weights from a YAML configuration file.
    Args: config_path: Path to data.yaml
        device: torch device
    Returns: Tensor of shape [num_classes] with class weights
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    weights_dict = cfg["class_weights"]

    # Ensure correct order by class index
    weights = [weights_dict[i] for i in sorted(weights_dict.keys())]

    return torch.tensor(weights, dtype=torch.float).to(device)
