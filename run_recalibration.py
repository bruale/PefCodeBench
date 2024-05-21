"""
Script to run the recalibration experiments
"""
# Author: Alessandro Brusaferri
# License: Apache-2.0 license

from tools.PrTSF_Recalib_tools import PrTsfRecalibEngine, load_data_model_configs


#--------------------------------------------------------------------------------------------------------------------
# Run recalibration configs
#--------------------------------------------------------------------------------------------------------------------
# Set PEPF task to execute
PF_task_name = 'DE_price'
# List of models setup to execute
setups_to_experiment = ['point-DNN', 'QR-DNN', 'JSU-DNN', 'STU-DNN', 'N-DNN']
# List of runs id (e.g., for ensemble components)
runs_id = ['recalib_opt_grid_1_1', 'recalib_opt_grid_1_2','recalib_opt_grid_1_3','recalib_opt_grid_1_4']
# Load hyperparams from file (select: load_tuned or optuna_tuner)
hyper_mode = 'load_tuned'

#---------------------------------------------------------------------------------------------------------------------
for exper_setup in setups_to_experiment:
    for run_id in runs_id:
        # Load experiments configuration from json file
        configs=load_data_model_configs(task_name=PF_task_name, exper_setup=exper_setup, run_id=run_id)

        # Instantiate recalibratione engine
        PrTSF_eng = PrTsfRecalibEngine(data_configs=configs['data_config'],
                                       model_configs=configs['model_config'])

        # Exec recalib loop over the test_set samples, using the tuned hyperparams
        test_predictions = PrTSF_eng.run_recalibration(hyper_mode=hyper_mode)

