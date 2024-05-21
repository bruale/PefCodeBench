"""
Script to execute post-processing routines (QRA, CP)
"""

# Author: Alessandro Brusaferri
# License: Apache-2.0 license

from tools.quant_proc_tools import QuantProc
import pickle
import os

#---------------------------------------------------------------------
# Configurations
#---------------------------------------------------------------------
# List of regions results to be post-processed
PF_tasks = ['DE_price', 'MGP_NORD', 'MGP_CNOR', 'MGP_CSUD', 'MGP_SUD', 'MGP_SARD', 'MGP_SICI']

# List of model configs to be post-processed
exper_configs=[
               'CP-DNN', 'QRA-DNN',
               'N-DNN','JSU-DNN','STU-DNN','QR-DNN',
               'CQ-N-DNN',  'CQ-JSU-DNN', 'CQ-STU-DNN','CQ-QR-DNN',
               'OCQ-N-DNN', 'OCQ-JSU-DNN', 'OCQ-STU-DNN', 'OCQ-QR-DNN'
               ]

# Run id to be post processed
run_id = 'recalib_opt_grid_1'
# Number of ensembles (run sub id) to be aggregated
num_ense = 4
# Experiment parameters (defined according to the recalibration runs configs)
num_cali_samples = 182
target_alpha = [0.2,0.4,0.6,0.8]
pred_horiz = 24

#----------------------------------------------------------------------------------------------------------------
# Process recalibration predictions by including ex-post qra and cp methods
#----------------------------------------------------------------------------------------------------------------
for PF_task in PF_tasks:
    exper_results = QuantProc(PF_task=PF_task,
                            experiments_to_analyse=exper_configs,
                            pred_horiz=pred_horiz,
                            run_id=run_id,
                            num_ense=num_ense,
                            num_cali_samples=num_cali_samples,
                            target_alpha=target_alpha).process_quantiles()

    task_results = {'results': exper_results,
                    'target_alpha': target_alpha,
                    'pred_horiz': pred_horiz}

    save_path = os.path.join(os.getcwd(), 'experiments', 'tasks', PF_task)
    with open(save_path + '/' + run_id + '_aggr_results.p', 'wb') as f:
        pickle.dump(task_results, f)