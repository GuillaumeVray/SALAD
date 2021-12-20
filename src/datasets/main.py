from .CXR import CXR_Dataset

def load_dataset(dataset_name, data_path, normal_class, isize):
    """Loads the dataset."""

    implemented_datasets = ('CXR_author')
    assert dataset_name in implemented_datasets

    dataset = None
        
    if dataset_name == 'CXR_author':
        dataset = CXR_Dataset(root=data_path, isize=isize)

    return dataset
