"""
Script to plot experiments results
"""
# Author: Alessandro Brusaferri
# License: Apache-2.0 license

from tools.results_analysis_tools import ExperAnalysis

#----------------------------------------------------------------------------------------------------------------
# Select PF task
#----------------------------------------------------------------------------------------------------------------
PF_task = 'MGP_NORD'
run_id = 'recalib_opt_grid_1'

#----------------------------------------------------------------------------------------------------------------
# Load experiments results and create the results analysis object
#----------------------------------------------------------------------------------------------------------------
exper_results = ExperAnalysis(PF_task=PF_task, run_id=run_id)

# ----------------------------------------------------------------------------------------------------------------
# Plot kupiec test
# ----------------------------------------------------------------------------------------------------------------
exper_results.plot_kupiec()

# ----------------------------------------------------------------------------------------------------------------
# Plot hourly PICP
# ----------------------------------------------------------------------------------------------------------------
# select the subset to plot
conf_to_plot=[
               #'QRA-DNN','CP-DNN','N-DNN','JSU-DNN','STU-DNN','QR-DNN',
               #'QRA-DNN','CQ-N-DNN', 'CQ-JSU-DNN', 'CQ-STU-DNN','CQ-QR-DNN',
               'QRA-DNN', 'OCQ-N-DNN', 'OCQ-JSU-DNN', 'OCQ-STU-DNN', 'OCQ-QR-DNN'
]
exper_results.plot_stepwise_PICP(conf_to_plot)

#----------------------------------------------------------------------------------------------------------------
# Execute DM test on pinball and winkler's score
#----------------------------------------------------------------------------------------------------------------
exper_results.plot_DM_test_pinball()
exper_results.plot_DM_test_winkler()
exper_results.plot_DM_test_mae()

#----------------------------------------------------------------------------------------------------------------
# Plot test Preds
#----------------------------------------------------------------------------------------------------------------
exper_results.plot_experiment_predictions('OCQR', PF_task)

# ----------------------------------------------------------------------------------------------------------------
# Print latex table of mean KPIs
# ----------------------------------------------------------------------------------------------------------------
print(exper_results.table_mean_kpis_latex())
print(exper_results.table_mean_kpis_markdown())
