import numpy as np
import math
import random
import time
import sys
import copy
import pdb
from datetime import datetime
from numpy.random import normal
from scipy.stats import norm
from scipy.optimize import fsolve
import argparse
from dataclasses import dataclass

################################################################################
# FORREAL BEGIN
################################################################################
import os
import pandas as pd
from meta_dict import dist2tid2eidmid, dist2absdiffstats, mid2absdiffstats

tid = os.environ.get('TASK_INDEX', 10)

dist2eidmid = {
    "bernoulli": dist2tid2eidmid["bernoulli"].get(str(tid), (0, 0)),
    "gauss": dist2tid2eidmid["gauss"].get(str(tid), (0, 0)),
}

class RealData:
    def __init__(self, eid, mid, reward_type):
        #
        self.eid, self.mid, self.reward_type = eid, mid, reward_type
        #
        if eid == 0 or mid == 0:
            self.valid = False
            return
        self.valid = True

        fname = f"{eid}_{mid}.csv"
        check_tid = os.getenv("TASK_INDEX")
        if check_tid is None:
            fname = f"~/Downloads/kdd2023_mab/real_data/{fname}"
        else:
            pass
        self.df = pd.read_csv(fname)
        tmp = self.df.groupby(["exptid", "groupid"]).count().reset_index()
        #
        self.gids, self.gid2uv= [], {}
        for i, row in tmp.iterrows():
            self.gids.append(row.groupid)
            self.gid2uv[str(row.groupid)] = row.uin
        sgids = sorted(self.gids)
        #
        self.control_gids, self.treat_gids = sgids[:2], sgids[2:]
        #
        self.armid2gids = {}
        self.armid2gids["0"] = self.control_gids
        for i, gid in enumerate(self.treat_gids):
            self.armid2gids[str(i + 1)] = [gid]
        #
        self.gid2armid = dict()
        for gid in self.control_gids:
            self.gid2armid[str(gid)] = 0
        for i, gid in enumerate(sorted(self.treat_gids)):
            self.gid2armid[str(gid)] = i + 1

        self.num_arm = 1 + len(self.treat_gids)
        self.df["armid"] = self.df['groupid'].apply(
                lambda x: self.gid2armid[str(x)])
        self.armid2df, self.armid2nextidx = dict(), dict()
        for armid, df in self.df.groupby("armid"):
            self.armid2df[str(armid)] = df
            self.armid2nextidx[str(armid)] = 0

        self.control_mean = self.armid2df["0"]["numerator"].mean()

        self.armid2mean, self.armid2var, self.armid2cnt = dict(), dict(), dict()
        for armid, df in self.armid2df.items():
            self.armid2cnt[armid] = df["numerator"].count()
        self.mincnt = min(self.armid2cnt.values())
        for armid in self.armid2df.keys():
            df = self.armid2df[armid][0:self.mincnt]
            self.armid2df[armid] = df.sample(frac=1, random_state=int(time.time())).reset_index(drop=True)
            self.armid2mean[armid] = df["numerator"].mean()
            self.armid2var[armid] = df["numerator"].var(ddof=1)
            self.armid2cnt[armid] = df["numerator"].count()
        self.means = list(self.armid2mean.values())
        self.vars = list(self.armid2var.values())
        self.cnts = list(self.armid2cnt.values())


        self.max_i, self.max_j = 0, 1
        if reward_type == "bernoulli":
            self.gaps, max_gap = [], 0
            for i in range(len(self.means)):
                for j in range(i + 1, len(self.means)):
                    cur_gap = abs(self.means[i] - self.means[j])
                    self.gaps.append(cur_gap)
                    if cur_gap > max_gap:
                        self.max_i, self.max_j = i, j
                        max_gap = cur_gap
        self.armidtrans = {
            "0": self.max_i,
            "1": self.max_j
        }

        ## GAP MEAN; GAP VAR
        if reward_type == "bernoulli":
            self.gap_mean, self.gap_var = abs(
                self.means[self.max_i] - self.means[self.max_j]), 0.05
        else:
            self.gaps = []
            for i in range(len(self.means)):
                for j in range(i + 1, len(self.means)):
                    self.gaps.append(abs(self.means[i] - self.means[j]))
            self.gap_mean = np.mean(self.gaps)
            self.gap_var = np.var(self.gaps)
        # self.gap_mean, self.gap_var = dist2absdiffstats[reward_type]
        # self.gap_mean, self.gap_var = mid2absdiffstats[str(mid)]

    def __str__(self) -> str:
        if not self.valid:
            return (
                f"eid={self.eid}, mid={self.mid}, \n"
                f"valid={self.valid}, \n"
            )
        return (
            f"eid={self.eid}, mid={self.mid}, \n"
            f"valid={self.valid}, \n"
            f"gids={self.gids}, \n"
            f"control_gids={self.control_gids}, \n"
            f"treat_gids={self.treat_gids}, \n"
            f"armid2gids={self.armid2gids}, \n"
            f"df.head() = {self.df.head()}, \n"
        )

    def return_reward_list(self, arm_id: int, number: int) -> np.ndarray:
        if self.reward_type == "bernoulli":
            arm_id = self.armidtrans[str(arm_id)]
        sidx = self.armid2nextidx[str(arm_id)]
        eidx = self.armid2nextidx[str(arm_id)] + number
        res = np.array(self.armid2df[str(arm_id)].iloc[sidx:eidx].numerator)
        self.armid2nextidx[str(arm_id)] = eidx
        return res

    def get_means(self):
        if self.reward_type == "bernoulli":
            return [self.means[self.max_i], self.means[self.max_j]]
        return self.means

    def get_vars(self):
        if self.reward_type == "bernoulli":
            return [self.vars[self.max_i], self.vars[self.max_j]]
        return self.vars

    def get_cnts(self):
        if self.reward_type == "bernoulli":
            return [self.cnts[self.max_i], self.cnts[self.max_j]]
        return self.cnts

    def p_values(self):
        res = []
        means, vars_, cnts = self.get_means(), self.get_vars(), self.get_cnts()
        for i in range(1, len(means)):
            std = np.sqrt(vars_[i] / cnts[i] + vars_[0] / cnts[0])
            p_value = 1 - norm.cdf(abs(means[i] - means[0]) / std)
            res.append(p_value)
        return res

################################################################################
# FORREAL END
################################################################################


def random_pick_id(id_list, id_prob):

        x = random.uniform(0,1)
        cumulative_probability=0.0
        for id, probability in zip(id_list,id_prob):
            cumulative_probability+=probability
            if x < cumulative_probability:
                final_id = id
                break
        return final_id


def compute_mu_sigma_of_Lognormal(mean,variance):
    sigma=(np.sqrt(np.log((variance/mean**2)+1)))
    mu=(np.log(mean)-sigma**2/2)

    return mu,sigma


def return_reward_list(type, number, mean, variance):
        if type == 'gauss':
            rewards = np.random.normal(loc= mean, scale=math.sqrt(variance), size=number)
        elif type == 'bernoulli':
            rewards = np.random.binomial(1,mean,number)
        elif type == 'lognormal':
            mu, sigma = compute_mu_sigma_of_Lognormal(mean, variance)

            rewards = np.random.lognormal(mu, sigma, number)

        return rewards



class Arm(object):
    def __init__(self, arm_id):
        self.arm_id = arm_id
        self.rewards = []

    def __str__(self):
        return (
            f"arm_id={self.arm_id}, rewards_mean={np.mean(self.rewards)}, "
            f"rewards_var: {np.var(self.rewards, ddof=1)}"
        )

    def get_mean(self):
        return np.mean(self.rewards)
    def get_var(self):
        return np.var(self.rewards, ddof=1)


class Environment(object):
    def __init__(
            self, Delta_min, bayesian_mean, bayesian_variance, arms,
            num_user, final_epoch, Exp, rd, stop_rule ='SPRT',
            sim=0, reward_type="bernoulli"):

        self.rd = rd

        self.arms = arms
        self.num_arm = len(arms)
        self.num_user=num_user

        self.prior_mean = np.zeros(shape=self.num_arm, dtype=np.float32)
        self.prior_covariance = np.diag(np.ones(shape = self.num_arm, dtype=np.float32)*(10000000000))
        self.find_opt = False


        # stop_rule: Bayes, BayesFactor
        self.stop_rule = stop_rule

        self.final_epoch = final_epoch

        self.bayesian_mean = bayesian_mean
        self.bayesian_variance = bayesian_variance

        self.Delta_min = Delta_min


        self.SPRT_likelihood_prior_ratio =  (norm.cdf(0, loc=self.bayesian_mean, scale=math.sqrt(self.bayesian_variance)) )/(1 - norm.cdf(self.Delta_min, loc=self.bayesian_mean, scale=math.sqrt(self.bayesian_variance)))
        self.opt = -1
        self.Return_type = -1

        self.exp_type = Exp

    def __str__(self):
        means = [arm.get_mean() for arm in self.arms]
        vars_ = [arm.get_var() for arm in self.arms]
        cnts = [len(arm.rewards) for arm in self.arms]
        p_values = []
        for i in range(1, len(means)):
            std = np.sqrt(vars_[i] / cnts[i] + vars_[0] / cnts[0])
            p_value = 1 - norm.cdf(abs(means[i] - means[0]) / std)
            p_values.append(p_value)
        return (
            f"rd: {self.rd.df.head()}, \n"
            f"arms_mean: {means}, arms_var: {vars}, cnts: {cnts}, p_values: {p_values}\n"
            f"num_arm={self.num_arm}, \n"
            f"num_user={self.num_user}, \n"
            f"prior_mean={self.prior_mean}, \n"
            f"prior_covariance={self.prior_covariance}, \n"
            f"find_opt={self.find_opt}, \n"
            f"stop_rule={self.stop_rule}, \n"
            f"final_epoch={self.final_epoch}, \n"
            f"bayesian_mean={self.bayesian_mean}, \n"
            f"bayesian_variance={self.bayesian_variance}, \n"
            f"Delta_min={self.Delta_min}, \n"
            f"SPRT_likelihood_prior_ratio={self.SPRT_likelihood_prior_ratio}, \n"
            f"opt={self.opt}, \n"
            f"Return_type={self.Return_type}, \n"
            f"exp_type={self.exp_type}, \n"
        )


    def generate_likelihood_data(self, final_chosen_prob, current_epoch, reward_type, mean, variance, version):
        total_cnt, total_reward = 0.0, 0.0
        # if current_epoch == 0 and version == 'online':
        #     for arm_id in range(self.num_arm):
        #         # print(reward_type, mean[arm_id], variance[arm_id])
        #         # new_reward_list = dist2rd[reward_type].return_reward_list(arm_id, 1)
        #         new_reward_list = self.rd.return_reward_list(arm_id, 1)
        #         if len(new_reward_list) > 0:
        #             total_cnt += len(new_reward_list)
        #             total_reward += new_reward_list.sum()
        #         # print(new_reward_list)
        #         self.arms[arm_id].rewards.extend( new_reward_list )
        # elif version == 'batch':
        if version == 'batch':
            for arm_id in range(self.num_arm):
                # print(final_chosen_prob, arm_id)
                num_users = int(round(final_chosen_prob[arm_id]*self.num_user,0))
                # new_reward_list = dist2rd[reward_type].return_reward_list(arm_id, num_users)
                new_reward_list = self.rd.return_reward_list(arm_id, num_users)
                if len(new_reward_list) > 0:
                    total_cnt += len(new_reward_list)
                    total_reward += new_reward_list.mean()
                # print(new_reward_list)
                self.arms[arm_id].rewards.extend( new_reward_list )
        elif version == 'online':
            arm_list = range(self.num_arm)
            for user in range(self.num_user):
                arm_id = random_pick_id(id_list=arm_list, id_prob = final_chosen_prob)
                # print(arm_id)
                # new_reward = dist2rd[reward_type].return_reward_list(arm_id, 1)
                new_reward = self.rd.return_reward_list(arm_id, 1)
                if len(new_reward) > 0:
                    total_cnt += len(new_reward)
                    total_reward += new_reward.mean()
                # print(new_reward)
                self.arms[arm_id].rewards.extend( new_reward )
        return total_cnt, total_reward

    # def generate_likelihood_data(self, final_chosen_prob, current_epoch, reward_type, mean, variance, version):
    #     msg = (
    #         f"final_chosen_prob={final_chosen_prob}, \n"
    #         f"current_epoch={current_epoch}, \n"
    #         f"reward_type={reward_type}, \n"
    #         f"mean={mean}, \n"
    #         f"variance={variance}, \n"
    #         f"version={version}, \n"
    #     )
    #     print(msg)

    #     if current_epoch == 0 and version == 'online':
    #         for arm_id in range(self.num_arm):
    #             # print(reward_type, mean[arm_id], variance[arm_id])
    #             new_reward_list = return_reward_list(reward_type,1, mean[arm_id], variance[arm_id] )
    #             self.arms[arm_id].rewards.extend( new_reward_list )
    #     elif version == 'batch':
    #         for arm_id in range(self.num_arm):
    #             num_users = int(round(final_chosen_prob[arm_id]*self.num_user,0))
    #             new_reward_list = return_reward_list(reward_type,num_users, mean[arm_id], variance[arm_id] )
    #             self.arms[arm_id].rewards.extend( new_reward_list )
    #     elif version == 'online':
    #         arm_list = range(self.num_arm)
    #         for user in range(self.num_user):
    #             arm_id = random_pick_id(id_list=arm_list, id_prob = final_chosen_prob)
    #             # print(arm_id)
    #             new_reward = return_reward_list(type=reward_type, number=1, mean=mean[arm_id], variance=variance[arm_id] )
    #             # print(new_reward)
    #             self.arms[arm_id].rewards.extend( new_reward )



    def t_test(self, real_variance, current_epoch):
        # multi-arm
        alpha = 0.05/(self.num_arm*(self.num_arm-1))
        post_mean = np.zeros(shape=self.num_arm, dtype=np.float32)
        post_var = np.zeros(shape=self.num_arm, dtype=np.float32)

        for arm_id in range(self.num_arm):
                total_data = len(self.arms[arm_id].rewards)
                post_mean[arm_id] = sum(self.arms[arm_id].rewards)/total_data
                post_var[arm_id] = real_variance[arm_id]/total_data


        if current_epoch == self.final_epoch-1:
            Flag = [ [-1 for arm in range(self.num_arm)] for arm in range(self.num_arm)]
            for arm_i in range(self.num_arm):
              for arm_j in range(arm_i+1, self.num_arm):

                standard_error = np.sqrt(post_var[arm_i] + post_var[arm_j])
                p_value = 1 - norm.cdf(abs(post_mean[arm_i]-post_mean[arm_j]) / standard_error)

                if p_value<alpha:
                    if post_mean[arm_i]>post_mean[arm_j]:
                        Flag[arm_i][arm_j] = 1
                        Flag[arm_j][arm_i] = 0

                    elif post_mean[arm_i]<post_mean[arm_j]:
                        Flag[arm_j][arm_i] = 1
                        Flag[arm_i][arm_j] = 0

            # if only unique optimal arm
            for arm_i in range(self.num_arm):
                if Flag[arm_i].count(1) == self.num_arm-1 and Flag[arm_i].count(-1)==1:
                    self.find_opt=True
                    self.opt = arm_i
                    self.Return_type = 1
                    break

        return post_mean, post_var





    def SPRT(self, Flag, real_variance, test_type, current_epoch):
            alpha, beta = 0.05, 0.2
            sub_alpha = alpha/(self.num_arm*(self.num_arm-1))
            sub_beta = beta/(self.num_arm*(self.num_arm-1))
            threshold_a, threshold_b = sub_beta/(1-sub_alpha), (1-sub_beta)/(sub_alpha)
            # threshold_a, threshold_b = (sub_beta/(1-sub_alpha)), ((1-sub_beta)/(sub_alpha))

            post_mean = np.zeros(shape=self.num_arm, dtype=np.float32)
            post_var = np.zeros(shape=self.num_arm, dtype=np.float32)

            for arm_id in range(self.num_arm):
                total_data = len(self.arms[arm_id].rewards)
                post_mean[arm_id] = sum(self.arms[arm_id].rewards)/total_data
                post_var[arm_id] = real_variance[arm_id]/total_data

            LogLikelihood = [[1 for j in range(self.num_arm)] for i in range(self.num_arm)]
            for arm_i in range(self.num_arm):
                for arm_j in range(self.num_arm):
                    if arm_j != arm_i and Flag[arm_i][arm_j]==-1:
                            total_var = post_var[arm_i]+post_var[arm_j]
                            mean_dif = post_mean[arm_i]-post_mean[arm_j]

                        # if self.exp_type == 'AB' and self.bayesian_variance > 0:
                            likeli_likeli_ratio_mean = (mean_dif*self.bayesian_variance+self.bayesian_mean*total_var)/(self.bayesian_variance+total_var)
                            likeli_likeli_ratio_var = (self.bayesian_variance*total_var)/(self.bayesian_variance+total_var)

                            # likeli_likeli_ratio = (norm.cdf(self.Delta_max, loc=likeli_likeli_ratio_mean, scale=math.sqrt(likeli_likeli_ratio_var))-norm.cdf(self.Delta_min, loc=likeli_likeli_ratio_mean, scale=math.sqrt(likeli_likeli_ratio_var)))/(norm.cdf(0, loc=likeli_likeli_ratio_mean, scale=math.sqrt(likeli_likeli_ratio_var))-norm.cdf(-self.Delta_max, loc=likeli_likeli_ratio_mean, scale=math.sqrt(likeli_likeli_ratio_var)))
                            # likeli_likeli_ratio = (1-norm.cdf(0, loc=likeli_likeli_ratio_mean, scale=math.sqrt(likeli_likeli_ratio_var)))/(norm.cdf(0, loc=likeli_likeli_ratio_mean, scale=math.sqrt(likeli_likeli_ratio_var)))


                            likeli_likeli_ratio = (1-norm.cdf(self.Delta_min, loc=likeli_likeli_ratio_mean, scale=math.sqrt(likeli_likeli_ratio_var)))/(norm.cdf(0, loc=likeli_likeli_ratio_mean, scale=math.sqrt(likeli_likeli_ratio_var)))
                            LogLikelihood[arm_i][arm_j] = (self.SPRT_likelihood_prior_ratio*likeli_likeli_ratio)

                            # likeli_likeli_ratio = np.exp(-(mean_dif-self.bayesian_mean)**2/(2*total_var))/np.exp(-mean_dif**2/(2*total_var))

                            # P_H_0 = np.exp(-mean_dif**2/(2*total_var))
                            # P_H_1 = np.exp(-(mean_dif-self.bayesian_mean)**2/(2*total_var))
                            # LogLikelihood[arm_i][arm_j] =  P_H_1/P_H_0

                            # LogLikelihood[arm_i][arm_j] = (1-norm.cdf(self.Delta_min, loc=mean_dif, scale=math.sqrt(total_var)))/(norm.cdf(0, loc=mean_dif, scale=math.sqrt(total_var)))


                        # elif self.exp_type == 'AA':
                        #     likeli_likeli_ratio = (1-norm.cdf(self.Delta_min, loc=mean_dif, scale=math.sqrt(total_var)))/(norm.cdf(self.Delta_min, loc=mean_dif, scale=math.sqrt(total_var)))
                        #     # # likeli_likeli_ratio = np.exp(-(mean_dif-self.bayesian_mean)**2/(2*total_var))/np.exp(-mean_dif**2/(2*total_var))
                        #     LogLikelihood[arm_i][arm_j] = np.log(likeli_likeli_ratio)

                        # elif self.bayesian_variance == 0.0:
                        #     # print("bayes variance = 0 when decide")
                        #     # likeli_likeli_ratio = np.exp(-(mean_dif-self.bayesian_mean)**2/(2*total_var))/np.exp(-mean_dif**2/(2*total_var))
                        #     likeli_likeli_ratio = (1-norm.cdf(self.Delta_min, loc=mean_dif, scale=math.sqrt(total_var)))/(norm.cdf(self.Delta_min, loc=mean_dif, scale=math.sqrt(total_var)))
                        #     # likeli_likeli_ratio = np.exp(-(mean_dif-self.bayesian_mean)**2/(2*total_var))/np.exp(-mean_dif**2/(2*total_var))
                        #     LogLikelihood[arm_i][arm_j] = np.log(likeli_likeli_ratio)


                            # if LogLikelihood[arm_i][arm_j]>= threshold_b:
                            #     Flag[arm_i][arm_j] = 1
                            # elif LogLikelihood[arm_i][arm_j]<= threshold_a:
                            #     Flag[arm_i][arm_j] = 0
                            if LogLikelihood[arm_i][arm_j]>=  threshold_b:
                                Flag[arm_i][arm_j] = 1
                            elif LogLikelihood[arm_i][arm_j]<= threshold_a:
                                Flag[arm_i][arm_j] = 0

            Return_type = -1
            Find_opt = False
            Opt = -1

            for arm_i in range(self.num_arm):
                    find_opt = False
                    if max(Flag[arm_i])>=1 and Flag[arm_i].count(-1)==1 and Flag[arm_i][arm_i]==-1:
                        find_opt = True
                        for arm_j in range(self.num_arm):
                            if arm_j!=arm_i and Flag[arm_i][arm_j]==0 and Flag[arm_j][arm_i]!=0:
                                find_opt = False
                                break

                    if find_opt == True:
                        Return_type = 1
                        Find_opt=True
                        Opt = arm_i
                        break

            items = []
            for arm_i in range(self.num_arm):
                    for arm_j in range(self.num_arm):
                        if arm_i != arm_j:
                            items.append(Flag[arm_i][arm_j])


            if Return_type != 1 and min(items)>=0:
                    if max(items)<=0:
                        Return_type = 0
                    elif max(items)==1:
                        Return_type = 2

            # return_type: 0 accept H0; 1 accept H1 and find the optimal arm; 2: accept H1 but fail to return an arm
            if test_type == "sequential":
                self.Return_type = Return_type
                self.find_opt = Find_opt
                self.opt = Opt

            elif test_type == "fixed" and current_epoch==self.final_epoch -1:
                self.Return_type = Return_type
                self.find_opt = Find_opt
                self.opt = Opt

            return Flag, post_mean, post_var




    # compute the cohurt according to the posterior distribution
    def compute_TS_final_assignment(self, post_mean, post_var ,fixed_ratio,  MC_simulation = 10000, post_type='gauss'):

        if post_type == 'beta':
            post_alpha = np.ones(shape=self.num_arm, dtype=np.float32)
            post_beta = np.ones(shape=self.num_arm, dtype=np.float32)
            for arm_id in range(self.num_arm):
                total_data = len(self.arms[arm_id].rewards)
                # mean = sum(self.arms[arm_id].rewards)/total_data
                post_alpha[arm_id] = 1+ sum(self.arms[arm_id].rewards)
                post_beta[arm_id] = 1+ total_data -sum(self.arms[arm_id].rewards)


        if fixed_ratio == 1:
            final_chosen_prob = [1/self.num_arm for arm in range(self.num_arm)]
        else:
            simulation_results = []
            for arm_id in range(len(post_mean)):
                if post_type=='gauss':
                    random = np.random.normal(loc= post_mean[arm_id], scale=math.sqrt( post_var[arm_id]), size=MC_simulation)
                elif post_type == 'beta':
                    random = np.random.beta(post_alpha[arm_id],post_beta[arm_id],size=MC_simulation)
                    # corresponding to alpha and beta parameter
                simulation_results.append(random)

            chosen_arm_index = np.argmax(simulation_results,axis=0)
            # print(chosen_arm_index)
            chosen_arm_index = chosen_arm_index.tolist()
            chosen_count = []
            for i in range(self.num_arm):
                chosen_count.append( chosen_arm_index.count(i) )

            chosen_prob= [arm_chosen_count/MC_simulation for arm_chosen_count in chosen_count]

            final_chosen_prob = [0 for arm in range(self.num_arm)]
            for arm in range(self.num_arm):
                final_chosen_prob[arm] = chosen_prob[arm]*(1-fixed_ratio)+fixed_ratio/self.num_arm


        return final_chosen_prob


    def compute_Eliminate_final_assignment(self, real_variance, Flag, assignBasedVar = 1):
        mean = [0 for arm in self.arms]

        for arm_id in range(self.num_arm):
                total_data = len(self.arms[arm_id].rewards)
                mean[arm_id] = sum(self.arms[arm_id].rewards)/total_data

        # compute_the_next_round chosen_prob
        Active_arm_index = [arm_id for arm_id in range(self.num_arm)]
        for arm_i in range(self.num_arm):
            for arm_j in range(self.num_arm):
                if Flag[arm_j][arm_i]==1:
                    Active_arm_index.remove(arm_i)
                    break

        chosen_prob = [0 for arm in self.arms]

        # 0: not based on variance; 1: based on real exact variance; 2: based on estimated variance
        if assignBasedVar == 0:
            for arm_i in range(self.num_arm):
                chosen_prob[arm_i] = 1/len(Active_arm_index) if arm_i in Active_arm_index else 0
        elif assignBasedVar == 1:
            total_real_var = 0
            for arm_i in range(self.num_arm):
                if arm_i in Active_arm_index:
                    total_real_var += real_variance[arm_i]

            for arm_i in range(self.num_arm):
                if arm_i in Active_arm_index:
                    chosen_prob[arm_i] = real_variance[arm_i]/total_real_var
        elif assignBasedVar == 2:
            total_estimated_var = 0
            estimated_var = [(sum([(reward - mean[arm_i])**2 for reward in self.arms[arm_i].rewards])/len(self.arms[arm_i].rewards)) for arm_i in range(self.num_arm)]
            for arm_i in range(self.num_arm):
                if arm_i in Active_arm_index:
                    total_estimated_var += estimated_var[arm_i]

            for arm_i in range(self.num_arm):
                if arm_i in Active_arm_index:
                    chosen_prob[arm_i] = estimated_var[arm_i]/total_estimated_var

        return chosen_prob


    def compute_UCB_final_assignment(self, current_epoch):
        mean = [0 for arm in self.arms]
        total_data = [0 for arm in self.arms]
        ucb = [0 for arm in self.arms]


        for arm_id in range(self.num_arm):
                total_data[arm_id] = len(self.arms[arm_id].rewards)
                mean[arm_id] = sum(self.arms[arm_id].rewards)/total_data[arm_id]
                ucb[arm_id] = mean[arm_id]+np.sqrt(2* np.log(current_epoch+1) / (total_data[arm_id]))

        chosen_prob = [0 for arm in self.arms]
        chosen_arm_index = np.argmax(ucb)
        chosen_prob[chosen_arm_index] = 1

        return chosen_prob






    def compute_EGreedy_final_assignment(self, epsilon):
        mean = [0 for arm in self.arms]
        total_data = [0 for arm in self.arms]


        for arm_id in range(self.num_arm):
                total_data[arm_id] = len(self.arms[arm_id].rewards)
                mean[arm_id] = sum(self.arms[arm_id].rewards)/total_data[arm_id]


        chosen_prob = [epsilon/self.num_arm for arm in self.arms]
        chosen_arm_index = np.argmax(mean)
        chosen_prob[chosen_arm_index] += 1-epsilon

        return chosen_prob




    def compute_InfoGreedy_final_assignment(self):
            means = [0 for arm in range(self.num_arm)]
            total_datas = [0 for arm in range(self.num_arm)]

            for arm_id in range(self.num_arm):
                total_datas[arm_id]=len(self.arms[arm_id].rewards)
                means[arm_id] = (sum(self.arms[arm_id].rewards)/total_datas[arm_id])

            if (means[0]*(1-means[0]))*(means[1]*(1-means[1])) == 0:
                final_chosen_prob = [0.5, 0.5]
            else:
                if total_datas[0]/total_datas[1]<math.sqrt((means[0]*(1-means[0]))/(means[1]*(1-means[1]))):
                    final_chosen_prob = [1, 0]
                elif total_datas[0]/total_datas[1]>math.sqrt((means[0]*(1-means[0]))/(means[1]*(1-means[1]))):
                    final_chosen_prob = [0, 1]
                else:
                    final_chosen_prob = [0.5, 0.5]

            return final_chosen_prob


    def compute_InfoRewardGreedy_final_assignment(self):
            means = [0 for arm in range(self.num_arm)]
            total_datas = [0 for arm in range(self.num_arm)]

            for arm_id in range(self.num_arm):
                total_datas[arm_id]=(len(self.arms[arm_id].rewards))
                means[arm_id] = (sum(self.arms[arm_id].rewards)/total_datas[arm_id])

            lambdas = total_datas[0]/total_datas[1]
            denom = (means[1]*(1-means[1]))
            if denom == 0:
                etas = 0
            else:
                etas = (means[0]*(1-means[0]))/ denom

            if (means[0]*(1-means[0]))*(means[1]*(1-means[1])) == 0:
                final_chosen_prob = [0.5, 0.5]
            else:
                if means[0] <= means[1]:
                    if lambdas<etas:
                        final_chosen_prob = [1, 0]
                    elif lambdas>etas:
                        final_chosen_prob =  [0, 1]
                    else:
                        final_chosen_prob = [0.5, 0.5]
                else:
                    if lambdas<etas:
                        final_chosen_prob = [0, 1]
                    elif lambdas>etas:
                        final_chosen_prob = [1, 0]
                    else:
                        final_chosen_prob = [0.5, 0.5]
            return final_chosen_prob



def run_simulations(args):
    # file name
    reward_type = args.reward_type # gauss, lognormal, bernoulli
    MAB_alg = args.MAB_alg # DATS, TS, Elimination
    total_user_each_arm = args.total_user_each_arm
    num_simulations = args.num_simulation
    num_epoch = args.num_epoch
    version = args.version
    stop_rule = args.stop_rule # Bayes; BayesFactor; SPRT
    test_type = args.test_type  # sequential, fixed
    epsilon = args.epsilon
    base_mean = args.base_mean
    fixed_ratio = args.TS_fixed_ratio
    # Delta_max = args.Delta_max
    Exp = args.Exp
    assignBasedVar = args.Elimination_type # 0: uniform; 1: exact variance; 2: estimated variance

    # filename = f"./{args.dir_name}/"
    filename = "./"

    filename += "_".join([
        "AB", "real", str(args.num_arm), str(num_epoch), str(version),
        str(total_user_each_arm), str(args.data_mincnt_frac), reward_type, test_type,
        MAB_alg, str(fixed_ratio), str(args.Delta_min_multiplier),
        str(datetime.now())]
    )

    filename = filename.replace(":", "")
    filename = filename.replace(" ", "_")
    f = open(filename, "w")

    sample_epochs = [] 
    sample_sizes = [] 
    total_sample_epochs = [] 
    total_sample_sizes = [] 
    find_truely_opts = 0  

    Accept_H0 = 0
    Accept_H1_and_FindOpt = 0
    Accept_H1_butNotOpt = 0

    averaged_rewards_allSims = []
    averaged_rewards_onlyRuns_allSims = []

    total_sim = len(dist2tid2eidmid[reward_type])

    for expand_id in range(args.expand_size):

        for sim in range(total_sim):

            # if sim != 10:
            #     continue
            # pdb.set_trace()

            eid, mid = dist2tid2eidmid[reward_type].get(str(sim), (0, 0))
            rd = RealData(eid, mid, reward_type)
            print((
                f"algo: {MAB_alg}; "
                f"sim: {sim + 1}/{len(dist2tid2eidmid[reward_type])}; "
                f"eid: {eid}; mid: {mid}; "
                f"num_arm: {rd.num_arm}; "
                f"num_groups: {len(rd.gids)}; "
                f"num_row: {len(rd.df)}; "
            ))

            # num_arm
            if MAB_alg in ["InfoRewardGreedy", "InfoGreedy"] or \
                    reward_type == "bernoulli":
                num_arm = 2
            else:
                num_arm = min(rd.num_arm, args.num_arm)

            # Delta_min
            Delta_min = rd.control_mean * args.Delta_min_multiplier
            bayesian_mean = rd.gap_mean
            bayesian_variance = rd.gap_var
            # bayesian_mean = args.bayesian_mean
            # bayesian_variance = args.bayesian_var

            # total_user_each_arm = 0
            # candi_tmp = rd.mincnt * args.data_mincnt_frac
            # if candi_tmp > args.total_user_each_arm:
            #     total_user_each_arm = args.total_user_each_arm
            # else:
            #     total_user_each_arm = min(args.total_user_each_arm, rd.mincnt)

            total_user_each_arm = min(
                    args.total_user_each_arm,
                    rd.mincnt * args.data_mincnt_frac)
            # if total_user_each_arm > rd.mincnt:
            #     continue

            num_user =int((num_arm*total_user_each_arm)/num_epoch)
            print(num_arm, total_user_each_arm, num_epoch, num_user)

            if reward_type == 'bernoulli' and MAB_alg=='TS':
                post_type = 'beta'
            else:
                post_type = 'gauss'

            mean = rd.get_means()
            variance = rd.get_vars()
            cnt_ = rd.get_cnts()

            pre_cumulative_cnt, pre_cumulative_reward = 0.0, 0.0
            f.write(f"\n-----------expand_id: {expand_id}; simulations: {sim}")
            f.write("\n[CURRENT EPOCH STATS]Real mean: "+str(mean))
            f.write("\n[CURRENT EPOCH STATS]Real Var"+str(variance))
            f.write("\n[CURRENT EPOCH STATS]Real Cnt"+str(cnt_))
            f.write("\n[CURRENT EPOCH STATS]Real p_values: "+str(rd.p_values()))

            arms = []
            for arm_id in range(num_arm):
                arms.append(Arm(arm_id=arm_id))

            final_chosen_prob = [1/num_arm for i in range(num_arm)]
            # Sim_Return_type = -1

            Flag = [[-1 for j in range(num_arm)] for i in range(num_arm)]

            env = Environment(arms= arms, num_user=num_user,  final_epoch = num_epoch, stop_rule=stop_rule, bayesian_mean=bayesian_mean,  bayesian_variance=bayesian_variance, Delta_min = Delta_min, Exp=Exp, rd=rd)

            epoch = 0
            for epoch in range(num_epoch):
                # f.write("\n------------epoch:"+str(epoch))

                t1 = time.time()
                cnt_this_epoch, reward_this_epoch = env.generate_likelihood_data(
                        final_chosen_prob=final_chosen_prob, current_epoch=epoch,
                        reward_type=reward_type, mean = mean, variance=variance, version= version)
                t2 = time.time()
                # pdb.set_trace()

                # reward_this_epoch = sum([final_chosen_prob[arm_id]*mean[arm_id] for arm_id in range(num_arm)])
                pre_cumulative_cnt += cnt_this_epoch
                pre_cumulative_reward += reward_this_epoch

                # if stop_rule == 'BayesFactor':
                #     Flag, post_mean, post_variance = env.BayesFactor(Flag=Flag, current_epoch = epoch, test_type = test_type, prior_ratio=prior_ratio, real_variance=variance)
                t3 = time.time()
                post_mean, post_variance = 0, 0
                if stop_rule == 'SPRT':
                    Flag, post_mean, post_variance = env.SPRT(Flag=Flag, current_epoch = epoch, test_type = test_type, real_variance=variance)
                elif stop_rule == "t-test":
                    # pdb.set_trace()
                    post_mean, post_variance = env.t_test(real_variance=variance, current_epoch = epoch)


                if MAB_alg == 'TS':
                    final_chosen_prob = env.compute_TS_final_assignment(post_mean=post_mean, post_var=post_variance, post_type=post_type, fixed_ratio = fixed_ratio)
                elif MAB_alg == 'Elimination':
                    final_chosen_prob = env.compute_Eliminate_final_assignment(real_variance=variance, Flag = Flag, assignBasedVar=assignBasedVar)
                elif MAB_alg == 'InfoGreedy':
                    final_chosen_prob = env.compute_InfoGreedy_final_assignment()
                    # f.write("\nPost Mean: "+str( post_mean))
                    # f.write("\nPost Var"+str(post_variance))
                    # f.write("\nfinal_chosen_prob: "+str(final_chosen_prob))
                    # f.write("\ntotal data: "+str(len(env.arms[0].rewards))+" " +str(len(env.arms[1].rewards)))
                elif MAB_alg=='InfoRewardGreedy':
                    final_chosen_prob = env.compute_InfoRewardGreedy_final_assignment()
                elif MAB_alg =='UCB':
                    final_chosen_prob = env.compute_UCB_final_assignment(current_epoch=epoch)
                elif MAB_alg == 'EGreedy':
                    final_chosen_prob = env.compute_EGreedy_final_assignment(epsilon=epsilon)
                else:
                    exit(1)
                t4 = time.time()

                # print(f"stop_rule: {stop_rule}; MAB_alg: {MAB_alg}; epoch: {epoch}; time cost: {t2-t1}, {t4-t3}")

                # pdb.set_trace()

                if env.Return_type == 0:
                    Accept_H0+=1
                elif env.Return_type == 1:
                    Accept_H1_and_FindOpt +=1
                    if Exp == "AB":
                        if mean[env.opt]==max(mean):
                            find_truely_opts+=1
                            sample_epochs.append(epoch+1)
                        sample_sizes.append(sum([len(arm.rewards) for arm in env.arms]))
                elif env.Return_type == 2:
                    Accept_H1_butNotOpt +=1

                if env.Return_type != -1:
                    # pdb.set_trace()
                    break

            total_sample_epochs.append(epoch+1)
            total_ss = sum([len(arm.rewards) for arm in env.arms])
            total_sample_sizes.append(total_ss)
            f.write(f"\n[CURRENT EPOCH STATS] Group sample sizes: "
                    f"{[len(arm.rewards) for arm in env.arms]}; "
                    f"Total sample size: {total_ss}")
            f.write("\n[ACCUM STATS]Find truely optimal:" + str(find_truely_opts) +
                    "; AcceptH1andFindOpt: " + str(Accept_H1_and_FindOpt) +
                    "; AcceptH0: " + str(Accept_H0) +
                    "; AcceptH1butNotFindOpt: " + str(Accept_H1_butNotOpt))
            # f.write("\nSum sample epochs before stop:"+str(sum(sample_epochs)) +
            # "; Sum total sample epochs:"+str(sum(total_sample_epochs)))


            remaining_cnt_this_sim, remaining_reward_this_sim = 0.0, 0.0
            if env.Return_type == 0:
                # accept H0
                remaining_cnt_this_sim = (num_epoch - epoch - 1) * num_user
                remaining_reward_this_sim = mean[0] * remaining_cnt_this_sim
            elif env.Return_type == 1 or env.Return_type == 2:
                # accept h1
                remaining_cnt_this_sim = (num_epoch - epoch - 1) * num_user
                remaining_reward_this_sim = mean[env.opt] * remaining_cnt_this_sim
            else:
                pass
            # if env.Return_type == 1:
            #     remaining_reward_this_sim = mean[env.opt]*(num_epoch-epoch-1)
            # elif env.Return_type == 0 or env.Return_type == 2:
            #     reward = sum([ (1/num_arm)*mean[arm] for arm in range(num_arm) ])
            #     remaining_reward_this_sim = reward*(num_epoch-epoch-1)
            # elif env.Return_type == -1:
            #     remaining_reward_this_sim = 0

            # # two types of averaged rewards, whether the remaining days count rewards
            total_cnt = pre_cumulative_cnt + remaining_cnt_this_sim
            total_reward = pre_cumulative_reward + remaining_reward_this_sim
            averaged_rewards_this_sim = total_reward / total_cnt
            averaged_rewards_onlyRuns_this_sim = pre_cumulative_reward / pre_cumulative_cnt
            # averaged_rewards_this_sim = (cumulative_reward + remaining_reward_this_sim )/num_epoch
            # averaged_rewards_onlyRuns_this_sim = (cumulative_reward )/(epoch+1)

            averaged_rewards_allSims.append(averaged_rewards_this_sim)
            averaged_rewards_onlyRuns_allSims.append(averaged_rewards_onlyRuns_this_sim)

            f.write(
                f"\n[CURRENT EPOCH STATS] Averaged rewards this sim: {averaged_rewards_this_sim}; "
                f"Averaged rewards Before Stop: {averaged_rewards_onlyRuns_this_sim}"
            )

            tmp = str(sample_sizes[-1]) if env.Return_type == 1 else 'None'
            f.write(
                f"\n[CURRENT EPOCH STATS] sample size when find truely opt: {tmp}; "
                f"total sample sizes before stop: {total_sample_sizes[-1]}; "
            )

            f.write(
                f"\n[ACCUM STATS] Averaged rewards all: {np.mean(averaged_rewards_allSims)}"
                f"Averaged rewards Before Stop: {np.mean(averaged_rewards_onlyRuns_allSims)}"
            )
            f.write(
                f"\n[ACCUM STATS]Power= {find_truely_opts/(sim+1 + total_sim * expand_id)}"
                f"; Averaged Sample size when find opt = {np.mean(sample_sizes)}"
                f"; Averaged Sample size before stop = {np.mean(total_sample_sizes)}"
            )
            f.flush()
    f.close()



if __name__ == '__main__':

    parse = argparse.ArgumentParser(description='Parameters')

    parse.add_argument('--num_epoch', default=20, type=int)

    parse.add_argument('--version', default='batch', type=str)

    parse.add_argument('--reward_type', default='bernoulli', type=str)

    parse.add_argument('--total_user_each_arm', default=10000, type=int)
    parse.add_argument('--data_mincnt_frac', default=0.9, type=float)
    # --test_type sequential(UCB ELimination EG TS+fixratio=0, TS+fixratio=1)
    # --test_type fixed (AB(TS+fixedratio=1) InfoGreedy InfoRewardGreedy)
    parse.add_argument('--test_type', default='sequential', type=str) #sequential, fixed

    parse.add_argument('--stop_rule', default='SPRT', type=str) #SPRT, t-test
    # UCB Elimination EG TS InfoGreedy InfoRewardGreedy
    parse.add_argument('--MAB_alg', default='Elimination', type=str) #Elimination, TS, InfoGreedy, InfoRewardGreedy

    parse.add_argument('--TS_fixed_ratio', default=1, type=float)  #0 (TS), 1 (AB)
    #
    parse.add_argument('--dir_name', type=str) #Elimination, TS, InfoGreedy, InfoRewardGreedy



    parse.add_argument('--Delta_min_multiplier', default=0.005, type=float)
  
    parse.add_argument('--bayesian_mean', default=0, type=float)
    parse.add_argument('--bayesian_var', default=0.02, type=float)

    parse.add_argument('--variance_list', nargs='+', type=float)


    parse.add_argument('--epsilon', default=0.1, type=float)
    parse.add_argument('--Elimination_type', default=1, type=int) #0, 1, 2; default 0
    parse.add_argument('--num_simulation', default=2000, type=int)
    parse.add_argument('--num_arm', default=5, type=int)
    parse.add_argument('--base_mean', default=0.2, type=float)
    parse.add_argument('--Delta_max', default=0.02, type=float)
    parse.add_argument('--Exp', default='AB', type=str) #AA, AB
    parse.add_argument('--expand_size', default=1, type=int)

    args = parse.parse_args()
    run_simulations(args)
