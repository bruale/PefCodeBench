"""
Definition of the keys employed by the data management utils
"""

# Author: Alessandro Brusaferri
# License: Apache-2.0 license

import os
columns_keys = {
    'Date': 'Date',
    'Hour': 'Hour',
    'idx_global': 'IDX__global',
    'idx_step': 'IDX__step',
    'idx_sub_step': 'IDX__sub_step'
}

#--------------------------------------------------------------------------------
# Features naming conventions employed to build the samples by moving windows
#--------------------------------------------------------------------------------
features_keys={
    # Employ just a single value related to the prediction step
    'const': 'CONST__',
    # Target variable
    'target': 'TARG__',
    # Employ just values included in the configured moving window till the step before predictions (i.e., lags)
    'past': 'PAST__',
    # Employ the series value related to the prediction step and eventual substeps (in case of multi-step predictions)
    'futu': 'FUTU__',
    # Specific features for the DE dataset
    'f_l-1': 'F_L-1__',
    'const_l-2': 'CONST_L-2__'
}


def get_dataset_save_path():
    return os.path.join(os.getcwd(), 'data', 'datasets')

