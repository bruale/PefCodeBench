"""
Utility functions for computing QRA and CP on recalibration results
"""

# Author: Alessandro Brusaferri
# License: Apache-2.0 license

import numpy as np
import pandas as pd
import os
import pickle

from tools.prediction_quantiles_tools import (fix_quantile_crossing,
                                              compute_qra, exec_cp, exec_cqr,
                                              build_alpha_quantiles_map,
                                              build_target_quantiles)
from tools.cp_pi import cts_pid

class QuantProc():
    """
    Exec QRA and CP
    """
    def __init__(self, PF_task, pred_horiz, experiments_to_analyse, run_id,
                 num_ense, num_cali_samples, target_alpha
                 ):

        # Store internal configs
        self.task_name = PF_task
        self.run_id = run_id
        self.num_ense = num_ense
        self.num_cali_samples = num_cali_samples
        self.pred_horiz = pred_horiz
        self.target_name = PF_task
        self.target_alpha = target_alpha
        self.target_quantiles = build_target_quantiles(self.target_alpha)
        self.alpha_q_map = build_alpha_quantiles_map(target_alpha=self.target_alpha, target_quantiles=self.target_quantiles)

        self.experiments_to_analyse = experiments_to_analyse


    def __get_exper_path__(self, exper_name, ense_comp):
        return os.path.join(os.getcwd(), 'experiments', 'tasks',
                            self.task_name, exper_name, self.run_id + '_' + str(ense_comp), 'results')

    def process_quantiles(self):
        def load_files(expe_n):
            results_e = []
            for e_c in range(self.num_ense):
                e_c = e_c + 1
                exper_save_path = self.__get_exper_path__(expe_n, ense_comp=e_c)
                results_file_name = '/recalib_test_results-tuned-grid_search.p'
                with open(exper_save_path + results_file_name, 'rb') as fp:
                    results_e.append(pickle.load(fp))
            return results_e
        # Load experiments results
        exper = {}
        for expe in self.experiments_to_analyse:
            if expe[:4]=='QRA-':
                model_n = 'point-' + expe[4:]
                results_e = load_files(model_n)
                exper[expe] = self.__compute_qra__(results_e, model_n)
            elif expe[:3]=='CP-':
                model_n = 'point-' + expe[3:]
                results_e = load_files(model_n)
                exper[expe] = self.__compute_cp__(results_e, model_n)
            elif expe[:3]=='CQ-':
                results_e = load_files(expe[3:])
                exper[expe] = self.__compute_cqr__(results_e)
            elif expe[:4]=='OCQ-':
                results_e = load_files(expe[4:])
                exper[expe] = self.__compute_pid__(results_e)
            else:
                results_e = load_files(expe)
                exper[expe] = self.__aggregate_quantiles__(results_e)
        return exper

    def __compute_qra__(self, results_e, model_n):
        file_path = os.path.join(os.getcwd(), 'experiments', 'tasks',self.task_name, model_n, self.run_id+'_qra.csv')
        try:
            aggr_df = pd.read_csv(file_path, index_col=0)
        except:
            # compute qra and cp from point preds, fix crossing
            ens_p=[]
            for e_c in range(self.num_ense):
                ens_p.append(results_e[e_c].loc[:,0.5].to_numpy().reshape(-1,1))
            ens_p=np.concatenate(ens_p, axis=1)
            ens_p_d = ens_p.reshape(-1, self.pred_horiz, ens_p.shape[-1])
            target_d = results_e[0].filter([self.target_name], axis=1).to_numpy().reshape(-1, self.pred_horiz)
            num_test_samples = ens_p_d.shape[0] - self.num_cali_samples
            test_PIs_qra=[]
            for t_s in range(num_test_samples):
                preds_cali = ens_p_d[t_s:self.num_cali_samples + t_s]
                preds_test = ens_p_d[self.num_cali_samples + t_s:self.num_cali_samples + t_s+1]
                y_cali = target_d[t_s:self.num_cali_samples + t_s]

                test_PIs_qra.append(compute_qra(preds_cali=preds_cali,
                                                y_cali=y_cali,
                                                preds_test=preds_test,
                                                settings={'target_quantiles': self.target_quantiles}))
                print(str(t_s))
            tes_PIs_qra=np.concatenate(test_PIs_qra, axis=0)
            aggr_df = results_e[0].filter([self.target_name], axis=1)
            aggr_df = aggr_df.iloc[self.pred_horiz * self.num_cali_samples:]
            for j in range(len(self.target_quantiles)):
                aggr_df[self.target_quantiles[j]]=tes_PIs_qra[:,j]
            aggr_df.to_csv(file_path)
        return aggr_df

    def __compute_cp__(self, results_e, model_n):
        file_path = os.path.join(os.getcwd(), 'experiments', 'tasks',self.task_name, model_n, self.run_id+'_cp.csv')
        try:
            aggr_df = pd.read_csv(file_path, index_col=0)
        except:
            # compute cp from point preds, fix crossing
            ens_p=[]
            for e_c in range(self.num_ense):
                ens_p.append(results_e[e_c].loc[:,0.5].to_numpy().reshape(-1,1))
            ens_p=np.mean(np.concatenate(ens_p, axis=1), axis=1)
            ens_p_d = ens_p.reshape(-1, self.pred_horiz, 1)
            target_d = results_e[0].filter([self.target_name], axis=1).to_numpy().reshape(-1, self.pred_horiz)
            num_test_samples = ens_p_d.shape[0] - self.num_cali_samples
            test_PIs=[]
            for t_s in range(num_test_samples):
                preds_cali = ens_p_d[t_s:self.num_cali_samples + t_s]
                preds_test = ens_p_d[self.num_cali_samples + t_s:self.num_cali_samples + t_s+1]
                y_cali = target_d[t_s:self.num_cali_samples + t_s]

                test_PIs.append(exec_cp(preds_cali=preds_cali,
                                                y_cali=y_cali,
                                                preds_test=preds_test,
                                                settings={'target_alpha': self.target_alpha}))

            test_PIs=np.concatenate(test_PIs, axis=0)
            aggr_df = results_e[0].filter([self.target_name], axis=1)
            aggr_df = aggr_df.iloc[self.pred_horiz * self.num_cali_samples:]
            for j in range(len(self.target_quantiles)):
                aggr_df[self.target_quantiles[j]]=test_PIs[:,j]
            aggr_df.to_csv(file_path)
        return aggr_df

    def __aggregate_quantiles__(self, results_e):
        aggr_df= results_e[0].filter([self.target_name], axis=1)
        # aggregate ensemble quantiles, fix crossimg, and compute cqr
        agg_q=[]
        for t_q in self.target_quantiles:
            q_c=[]
            for e_c in range(self.num_ense):
                q_c.append(results_e[e_c].loc[:,t_q].to_numpy().reshape(-1,1))
            agg_q.append(np.mean(np.concatenate(q_c, axis=1), axis=1).reshape(-1,1))
        q_ens=fix_quantile_crossing(np.concatenate(agg_q, axis=1))
        for j in range(len(self.target_quantiles)):
            aggr_df[self.target_quantiles[j]]=q_ens[:,j]
        return aggr_df.iloc[self.pred_horiz*self.num_cali_samples:]

    def __compute_cqr__(self, results_e):
        # aggregate ensemble quantiles, fix crossimg, and compute cqr
        agg_q = []
        for t_q in self.target_quantiles:
            q_c = []
            for e_c in range(self.num_ense):
                q_c.append(results_e[e_c].loc[:, t_q].to_numpy().reshape(-1, 1))
            agg_q.append(np.mean(np.concatenate(q_c, axis=1), axis=1).reshape(-1, 1))
        q_ens = fix_quantile_crossing(np.concatenate(agg_q, axis=1))

        target_d = results_e[0].filter([self.target_name], axis=1).to_numpy().reshape(-1, self.pred_horiz, )
        preds_d = q_ens.reshape(-1, self.pred_horiz, q_ens.shape[-1])
        num_test_samples = preds_d.shape[0] - self.num_cali_samples
        settings={'target_alpha': self.target_alpha,
                  'cp_options': {'cqr_mode': 'asym'},
                  'q_alpha_map': build_alpha_quantiles_map(target_quantiles=self.target_quantiles,
                                                           target_alpha=self.target_alpha)
        }
        test_PIs = []
        for t_s in range(num_test_samples):
            preds_cali = preds_d[t_s:self.num_cali_samples + t_s]
            preds_test = preds_d[self.num_cali_samples + t_s:self.num_cali_samples + t_s + 1]
            y_cali = target_d[t_s:self.num_cali_samples + t_s]

            test_PIs.append(exec_cqr(preds_cali=preds_cali,
                                     y_cali=y_cali,
                                     preds_test=preds_test,
                                     settings=settings))

        test_PIs = np.concatenate(test_PIs, axis=0)
        aggr_df = results_e[0].filter([self.target_name], axis=1)
        aggr_df = aggr_df.iloc[self.pred_horiz * self.num_cali_samples:]
        for j in range(len(self.target_quantiles)):
            aggr_df[self.target_quantiles[j]] = test_PIs[:, j]
        return aggr_df

    def __compute_pid__(self, results_e, lr=0.01,  KI = 10, T_burnin=7, Tin = 1e9, delta=5e-2):
        # aggregate ensemble quantiles, fix crossimg, and compute cqr
        agg_q = []
        for t_q in self.target_quantiles:
            q_c = []
            for e_c in range(self.num_ense):
                q_c.append(results_e[e_c].loc[:, t_q].to_numpy().reshape(-1, 1))
            agg_q.append(np.mean(np.concatenate(q_c, axis=1), axis=1).reshape(-1, 1))
        q_ens = fix_quantile_crossing(np.concatenate(agg_q, axis=1))

        preds_d = q_ens.reshape(-1, self.pred_horiz, q_ens.shape[-1])

        q_alpha_map = build_alpha_quantiles_map(target_quantiles=self.target_quantiles,
                                                target_alpha=self.target_alpha)

        df = results_e[0].filter([self.target_name], axis=1)

        # initialize the output quantiles with the median
        pred_q={q_alpha_map['med']: q_ens[:, q_alpha_map['med']]}
        for alpha in self.target_alpha:
            # get index of the lower/upper quantiles for the current alpha from the map
            lq_idx = q_alpha_map[alpha]['l']
            uq_idx = q_alpha_map[alpha]['u']
            sets_h=[]
            for h in range(self.pred_horiz):
                t_s = str(h)
                t_0 = t_s + ':00:00'
                t_1 = t_s + ':00:30'
                df_h=df.between_time(t_0, t_1).copy()
                df_h.rename(columns={self.target_name:'y'}, inplace=True)
                df_h['forecasts'] = [np.array([preds_d[j, h, lq_idx], preds_d[j, h, uq_idx]])
                                        for j in range(len(df_h))]

                Csat=(2/np.pi)*(np.ceil(np.log(Tin)*delta)-(1/np.log(Tin)))

                results = cts_pid(data=df_h, alpha=alpha, lr=lr, Csat=Csat, KI=KI, T_burnin=T_burnin)
                sets_h.append(np.stack(results['sets'], axis=1).T)

            sets_p = np.stack(sets_h, axis=1).reshape(-1, 2)
            pred_q[q_alpha_map[alpha]['l']] = sets_p[:, 0]
            pred_q[q_alpha_map[alpha]['u']] = sets_p[:, 1]

        test_q = dict(sorted(pred_q.items()))
        test_q = np.array(list(test_q.values())).T
        test_q = fix_quantile_crossing(test_q)
        results_df = results_e[0].filter([self.target_name], axis=1)
        for j in range(len(self.target_quantiles)):
            results_df[self.target_quantiles[j]] = test_q[:, j]
        return results_df.iloc[self.pred_horiz * self.num_cali_samples:]