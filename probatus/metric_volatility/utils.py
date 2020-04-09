import numpy as np


def sample_data(X, y, sampling_type, sampling_fraction, dataset_name='dataset'):
    check_sampling_input(sampling_type, sampling_fraction, dataset_name)

    if sampling_type is None:
        return X, y

    number_of_samples = np.ceil(sampling_fraction * X.shape[0]).astype(int)
    array_index = list(range(X.shape[0]))

    if sampling_type == 'bootstrap':
        rows_indexes = np.random.choice(array_index, number_of_samples, replace=True)
    else:
        if sampling_fraction == 1 or number_of_samples == X.shape[0]:
            return X,y
        else:
            rows_indexes = np.random.choice(array_index, number_of_samples, replace=True)
    return X[rows_indexes], y[rows_indexes]


def check_sampling_input(sampling_type, fraction, dataset_name):
    if sampling_type is not None:
        if sampling_type == 'bootstrap':
            if fraction <= 0:
                raise(ValueError(f'For bootstrapping {dataset_name} fraction needs to be above 0'))
        elif sampling_type == 'subsample':
            if  fraction <= 0 or fraction >= 1:
                raise(ValueError(f'For bootstrapping {dataset_name} fraction needs to be be above 0 and below 1'))
        else:
            raise(ValueError('This sampling method is not implemented'))