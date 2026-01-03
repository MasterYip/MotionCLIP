from .amass import AMASS, G1AMASS

def get_dataset(name="amass"):
    """
    Get dataset class by name.
    
    Args:
        name: Dataset name ('amass' or 'g1_amass')
        
    Returns:
        Dataset class
        
    Note:
        Both 'amass' and 'g1_amass' use the same AMASS class.
        G1 mode is controlled by the 'use_g1' parameter passed to the dataset.
    """
    if name in ["amass", "g1_amass"]:
        return AMASS
    raise ValueError(f"Unknown dataset: {name}")


def get_datasets(parameters, clip_preprocess, split="train"):
    """
    Initialize train and test datasets.
    
    For G1 datasets (dataset='g1_amass'), automatically sets use_g1=True.
    """
    DATA = AMASS
    
    # Auto-enable G1 mode if dataset is g1_amass
    if parameters.get('dataset') == 'g1_amass':
        parameters['use_g1'] = True
        DATA = G1AMASS

    if split == 'all':
        train = DATA(split='train', clip_preprocess=clip_preprocess, **parameters)
        test = DATA(split='vald', clip_preprocess=clip_preprocess, **parameters)

        # add specific parameters from the dataset loading
        train.update_parameters(parameters)
        test.update_parameters(parameters)
    else:
        dataset = DATA(split=split, clip_preprocess=clip_preprocess, **parameters)
        train = dataset

        # test: shallow copy (share the memory) but set the other indices
        from copy import copy
        test = copy(train)
        test.split = test

        # add specific parameters from the dataset loading
        dataset.update_parameters(parameters)

    datasets = {"train": train,
                "test": test}

    return datasets
