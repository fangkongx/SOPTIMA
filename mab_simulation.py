import numpy as np
import math
import random
import sys
from datetime import datetime
from scipy.stats import norm 
from scipy.optimize import fsolve
import argparse

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
       
    
class Environment(object):
    def __init__(self, Delta_min, bayesian_mean, bayesian_variance, arms, num_user, final_epoch, Exp, stop_rule ='SPRT'):
        self.arms = arms
        self.num_arm = len(arms)
        
        self.num_user = num_user
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
      

    def generate_likelihood_data(self, final_chosen_prob, current_epoch, reward_type, mean, variance, version):
        if current_epoch == 0 and version == 'online':
            for arm_id in range(self.num_arm):
                # print(reward_type, mean[arm_id], variance[arm_id])
                new_reward_list = return_reward_list(reward_type,1, mean[arm_id], variance[arm_id] )
                self.arms[arm_id].rewards.extend( new_reward_list )
        elif version == 'batch':
            for arm_id in range(self.num_arm):
                num_users = int(round(final_chosen_prob[arm_id]*self.num_user,0))
                new_reward_list = return_reward_list(reward_type,num_users, mean[arm_id], variance[arm_id] )
                self.arms[arm_id].rewards.extend( new_reward_list )
        elif version == 'online':
            arm_list = range(self.num_arm)
            for user in range(self.num_user):
                arm_id = random_pick_id(id_list=arm_list, id_prob = final_chosen_prob)
                # print(arm_id)
                new_reward = return_reward_list(type=reward_type, number=1, mean=mean[arm_id], variance=variance[arm_id] )
                # print(new_reward)
                self.arms[arm_id].rewards.extend( new_reward )
    


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
            etas = (means[0]*(1-means[0]))/(means[1]*(1-means[1]))

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
    filename = "./Exp_0106_multiArm/"

    num_simulations = args.num_simulation
    num_epoch = args.num_epoch
    fixed_ratio = args.TS_fixed_ratio
    num_arm = args.num_arm
    version = args.version

    base_mean = args.base_mean
    Delta_min = args.Delta_min
    # Delta_max = args.Delta_max

    bayesian_mean = args.bayesian_mean
    bayesian_variance = args.bayesian_var

    Exp = args.Exp
    
    assignBasedVar = args.Elimination_type # 0: uniform; 1: exact variance; 2: estimated variance
    
    stop_rule = args.stop_rule # Bayes; BayesFactor; SPRT
    MAB_alg = args.MAB_alg # DATS, TS, Elimination
    reward_type = args.reward_type # gauss, lognormal, bernoulli
    test_type = args.test_type  # sequential, fixed
    total_user_each_arm = args.total_user_each_arm

    epsilon = args.epsilon
    

    num_user =int((num_arm*total_user_each_arm  )/num_epoch) 

   
    if MAB_alg == 'TS':
        filename = filename+Exp+"Sim_"+str(num_simulations)+"Epoch_"+str(num_epoch)+'_K'+str(num_arm)+'_'+MAB_alg+"_Ratio_"+str(fixed_ratio)+'_BaseMean'+str(base_mean)+'_DeltaMean'+str(bayesian_mean)+'_DeltaVar'+str(bayesian_variance)+'_Min'+str(Delta_min)+'_'+test_type+'_'+stop_rule+'_'+reward_type+'_'+"N_"+str(total_user_each_arm)+'_' +"version_"+version+'_'
    elif MAB_alg == 'Elimination':
        filename = filename+Exp+"Sim_"+str(num_simulations)+"Epoch_"+str(num_epoch)+'_K'+str(num_arm)+'_'+MAB_alg+"_EliType_"+str(assignBasedVar)+'_BaseMean'+str(base_mean)+'_DeltaMean'+str(bayesian_mean)+'_DeltaVar'+str(bayesian_variance)+'_Min'+str(Delta_min)+'_'+test_type+'_'+stop_rule+'_'+reward_type+'_'+"N_"+str(total_user_each_arm)+'_' +"version_"+version+'_'
    elif MAB_alg == 'InfoGreedy' or MAB_alg == 'InfoRewardGreedy':
        filename = filename+Exp+"Sim_"+str(num_simulations)+"Epoch_"+str(num_epoch)+'_K'+str(num_arm)+'_'+MAB_alg+'_BaseMean'+str(base_mean)+'_DeltaMean'+str(bayesian_mean)+'_DeltaVar'+str(bayesian_variance)+'_Min'+str(Delta_min)+'_'+test_type+'_'+stop_rule+'_'+reward_type+'_'+"N_"+str(total_user_each_arm)+'_' +"version_"+version+'_'
    elif MAB_alg == 'UCB':
        filename = filename+Exp+"Sim_"+str(num_simulations)+"Epoch_"+str(num_epoch)+'_K'+str(num_arm)+'_'+MAB_alg+'_BaseMean'+str(base_mean)+'_DeltaMean'+str(bayesian_mean)+'_DeltaVar'+str(bayesian_variance)+'_Min'+str(Delta_min)+'_'+test_type+'_'+stop_rule+'_'+reward_type+'_'+"N_"+str(total_user_each_arm)+'_' +"version_"+version+'_'
    elif MAB_alg == 'EGreedy':
        filename = filename+Exp+"Sim_"+str(num_simulations)+"Epoch_"+str(num_epoch)+'_K'+str(num_arm)+'_'+MAB_alg+"_Epsilon_"+str(epsilon)+'_BaseMean'+str(base_mean)+'_DeltaMean'+str(bayesian_mean)+'_DeltaVar'+str(bayesian_variance)+'_Min'+str(Delta_min)+'_'+test_type+'_'+stop_rule+'_'+reward_type+'_'+"N_"+str(total_user_each_arm)+'_' +"version_"+version+'_'
    
    filename += str(datetime.now())
    filename = filename.replace(":", "")
    f = open(filename, "w")

    if reward_type == 'bernoulli' and MAB_alg=='TS':
        post_type = 'beta'
    else:
        post_type = 'gauss'
    
   
   
    sample_epochs = []
    total_sample_epochs = []
    find_truely_opts = 0

    Accept_H0 = 0
    Accept_H1_and_FindOpt = 0
    Accept_H1_butNotOpt = 0

    averaged_rewards_allSims = []
    averaged_rewards_onlyRuns_allSims = []
    
    for sim in range(num_simulations):
        if Exp == 'AB' and bayesian_variance>0 and num_arm == 2:
            mean = [base_mean]
            for i in range(1,num_arm):
                gap = np.random.normal(loc= bayesian_mean, scale=math.sqrt(bayesian_variance), size=1)[0]
                value = base_mean+gap
                while (reward_type== 'bernoulli' and value<0) or (reward_type== 'bernoulli' and value>1):
                    gap = np.random.normal(loc= bayesian_mean, scale=math.sqrt(bayesian_variance/2), size=1)[0]
                    value = base_mean+gap
                mean.append(value)
        elif Exp == 'AB' and bayesian_variance>0 and num_arm > 2:
            mean = [base_mean]
            for i in range(1,num_arm):
                gap = np.random.normal(loc= bayesian_mean, scale=math.sqrt(bayesian_variance/2), size=1)[0]
                value = base_mean+gap
                while (reward_type== 'bernoulli' and value<0) or (reward_type== 'bernoulli' and value>1):
                    gap = np.random.normal(loc= bayesian_mean, scale=math.sqrt(bayesian_variance/2), size=1)[0]
                    value = base_mean+gap
                mean.append(value)
        elif Exp == 'AA':
            mean = [base_mean for arm in range(num_arm)]
        elif bayesian_variance == 0.0 and num_arm==2:
            # 2-arm case with known delta
            mean = [base_mean, base_mean+bayesian_mean]

        


        if reward_type == 'bernoulli':
            variance =[round(mean[i]*(1-mean[i]),4) for i in range(num_arm)]
        else:
            variance =args.variance_list
            if len(variance)<num_arm:
                variance.extend([variance[-1] for i in range(num_arm - len(variance))])



        cumulative_reward = 0
        f.write("\n-------------simulations:"+str(sim))
        f.write("\nReal mean: "+str(mean))
        f.write("\nReal Var"+str(variance))
       
        arms = []
        for arm_id in range(num_arm):
            arms.append(Arm(arm_id=arm_id))
    
        final_chosen_prob = [1/num_arm for i in range(num_arm)]
        # Sim_Return_type = -1

        Flag = [[-1 for j in range(num_arm)] for i in range(num_arm)]

        env = Environment(arms= arms, num_user=num_user,  final_epoch = num_epoch, stop_rule=stop_rule, bayesian_mean=bayesian_mean,  bayesian_variance=bayesian_variance, Delta_min = Delta_min, Exp=Exp)
        
        for epoch in range(num_epoch):
            # f.write("\n------------epoch:"+str(epoch))

            env.generate_likelihood_data(final_chosen_prob=final_chosen_prob,current_epoch=epoch, reward_type=reward_type, mean = mean, variance=variance, version= version)

            
            reward_this_epoch = sum([final_chosen_prob[arm_id]*mean[arm_id] for arm_id in range(num_arm)])
            cumulative_reward+=reward_this_epoch

            # if stop_rule == 'BayesFactor':
            #     Flag, post_mean, post_variance = env.BayesFactor(Flag=Flag, current_epoch = epoch, test_type = test_type, prior_ratio=prior_ratio, real_variance=variance)
            if stop_rule == 'SPRT':
                Flag, post_mean, post_variance = env.SPRT(Flag=Flag, current_epoch = epoch, test_type = test_type, real_variance=variance)
            elif stop_rule == "t-test":
                post_mean, post_variance = env.t_test(real_variance=variance, current_epoch = epoch)
            
            

            



            # if env.find_opt == True:
            #     Find_opt = env.opt
            #     # f.write("\nFind_opt "+str(Find_opt))
            #     sum_stop_simulations+=1
            #     if Exp == "AB":
            #         if mean[Find_opt]==max(mean):
            #             find_truely_opts+=1
            #         sample_epochs.append(epoch+1)
            #     break
            
            
           
            # f.write("\nreward_this_epoch: "+str(reward_this_epoch))
            # f.write("\nPost Mean: "+str( post_mean))
            # f.write("\nPost Var"+str(post_variance))
            # f.write("\nFlag: "+str(Flag))
            # f.write("\nfinal_chosen_prob: "+str(final_chosen_prob))


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
                

            if env.Return_type == 0:
                Accept_H0+=1
            elif env.Return_type == 1:
                Accept_H1_and_FindOpt +=1
                if Exp == "AB":
                    if mean[env.opt]==max(mean):
                        find_truely_opts+=1
                        sample_epochs.append(epoch+1)
            elif env.Return_type == 2:
                Accept_H1_butNotOpt +=1

            if env.Return_type != -1:
                break
        
        total_sample_epochs.append(epoch+1)
        f.write("\nFind truely optimal:"+str(find_truely_opts)+"; AcceptH1andFindOpt: "+str(Accept_H1_and_FindOpt)+"; AcceptH0: "+str(Accept_H0)+"; AcceptH1butNotFindOpt: "+str(Accept_H1_butNotOpt))
        f.write("\nSum sample epochs before stop:"+str(sum(sample_epochs)) +"; Sum total sample epochs:"+str(sum(total_sample_epochs)))
       

        # Reward: later update 

        if env.Return_type == 1:
            remaining_reward_this_sim = mean[env.opt]*(num_epoch-epoch-1)
        elif env.Return_type == 0 or env.Return_type == 2:
            reward = sum([ (1/num_arm)*mean[arm] for arm in range(num_arm) ])
            remaining_reward_this_sim = reward*(num_epoch-epoch-1)
        elif env.Return_type == -1:
            remaining_reward_this_sim = 0

        # # two types of averaged rewards, whether the remaining days count rewards
        averaged_rewards_this_sim = (cumulative_reward + remaining_reward_this_sim )/num_epoch
        averaged_rewards_onlyRuns_this_sim = (cumulative_reward )/(epoch+1)

        averaged_rewards_allSims.append(averaged_rewards_this_sim)
        averaged_rewards_onlyRuns_allSims.append(averaged_rewards_onlyRuns_this_sim)
    
        f.write("\nAveraged rewards all: "+str(sum(averaged_rewards_allSims)/len(averaged_rewards_allSims))+"; Averaged rewards Before Stop: "+str(sum(averaged_rewards_onlyRuns_allSims)/len(averaged_rewards_onlyRuns_allSims)))
        # f.write("; Averaged rewards Before Stop: "+str(sum(averaged_rewards_onlyRuns_allSims)/len(averaged_rewards_onlyRuns_allSims)))
        
        if (sim+1)%1000 == 0 and sim!=0:
            f.write("\nPower="+ str(find_truely_opts/(sim+1)) + "; Averaged Sample size = "+ str(sum(sample_epochs)/find_truely_opts)+ "; Averaged user observations per arm = "+ str(sum(sample_epochs)*num_user/(find_truely_opts*num_arm)))
        

        f.flush()
    f.close()














if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Parameters') 
    parse.add_argument('--num_simulation', default=2000, type=int)
    parse.add_argument('--num_epoch', default=20, type=int)
    parse.add_argument('--num_arm', default=5, type=int)
    parse.add_argument('--version', default='batch', type=str)

    parse.add_argument('--base_mean', default=0.2, type=float)
    parse.add_argument('--Delta_min', default=0.02, type=float)
    # parse.add_argument('--Delta_max', default=0.02, type=float)
    parse.add_argument('--bayesian_mean', default=0, type=float)
    parse.add_argument('--bayesian_var', default=0.02, type=float)

    parse.add_argument('--reward_type', default='bernoulli', type=str) 
    parse.add_argument('--total_user_each_arm', default=10000, type=int)


    parse.add_argument('--Exp', default='AA', type=str) #AA, AB
    
    parse.add_argument('--test_type', default='sequential', type=str) #sequential, fixed
    parse.add_argument('--stop_rule', default='SPRT', type=str) #SPRT, t-test
    parse.add_argument('--MAB_alg', default='Elimination', type=str) #Elimination, TS, InfoGreedy, InfoRewardGreedy
    parse.add_argument('--Elimination_type', default=1, type=int) #0, 1, 2; default 0
    parse.add_argument('--TS_fixed_ratio', default=1, type=float)  #0 (TS), 1 (AB)
    parse.add_argument('--epsilon', default=0.1, type=float)  

    parse.add_argument('--variance_list', nargs='+', type=float)
        
    args = parse.parse_args() 



    run_simulations(args)






