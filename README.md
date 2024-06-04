# PefCodeBench

Codebase to reproduce the experiments performed in the paper (pre-print, under review):

- Brusaferri, A., Ballarino, A., Grossi, L., & Laurini, F. (2024). On-line conformalized neural networks ensembles for probabilistic forecasting of day-ahead electricity prices, https://arxiv.org/abs/2404.02722

---

### Getting started

The results of the experiments are stored as pickle files within the <code>experiments/task</code> folder, 
aggregated by <code>regions-> methods-> recalibration</code> runs. 

Note: in the code we used 'DE' to label the German market while in the paper we employed 'GE' since 'DE' was already employed to label Deep Ensembles.

The employed packages versions are stored in the <code>requirements.txt</code> file (Python 3.8.10). Besides, the code of DM-tests, Kupiec tests, Distributional NNs and Conformal PI are built upon https://github.com/jeslago/epftoolbox, https://github.com/rafa-rod/vartests, https://github.com/gmarcjasz/distributionalnn and https://github.com/aangelopoulos/conformal-time-series respectively.  

The script <code>results_analysis.py</code> contains the functions to obtain the plots/tables in the *Results* section of the paper from the stored pickle.

To execute the recalibration experiments from scratch (i.e., models re-training), open the <code>run_recalibration.py</code> script,
select the dataset to execute in the <code>PF_task_name</code> variable and run the script.
Specific local minima reached by the training algorithm may lead to fluctuations in the test predictions (e.g., QR vs JSU vs Stu). 
Still, the implemented Conformal Prediction based techniques are expected to improve the hourly calibration of the backbone models across the different settings. 

The <code>run_recalibration.py</code> script will store the experiments results in the related <code>results</code> subfolders.
Create a copy of the <code>recalib_opt_grid_1_(1-4)</code> folders by defining a new name (e.g., <code>my_recalib_opt_grid_1_(1-4)</code>) before running the experiments 
to keep the original experiments results, otherwise they will be updated during each run of the script.

Keep the variable <code>hyper_mode</code> set to  <code>'load_tuned'</code> to load the stored hyperparameters values.
Set it to <code>'optuna_tuner'</code> for executing also hyperparameter search from scratch.

Once the recalibration runs are completed, run the <code>exec_qra_cp.py</code> script 
to execute the post-processing routines (i.e., Quantile Regression Averaging and Conformal Prediction).

If you created your own experimental copy, assign the chosen name to the <code>run_id</code> variable within both <code>run_recalibration.py</code>, <code>exec_qra_cp.py</code> and  <code>results_analysis.py</code> 

---

### Citation

If you use this code in your publication, please cite our paper:
https://arxiv.org/abs/2404.02722

      @misc{brusaferri2024online,
      title={On-line conformalized neural networks ensembles for probabilistic forecasting of day-ahead electricity prices}, 
      author={Alessandro Brusaferri and Andrea Ballarino and Luigi Grossi and Fabrizio Laurini},
      year={2024},
      eprint={2404.02722},
      archivePrefix={arXiv},
      primaryClass={cs.LG}}
