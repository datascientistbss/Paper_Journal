import numpy as np
import pandas as pd
from scipy.stats import norm, poisson, nbinom, skellam
#import sys
#print(sys.path)
#sys.path.insert(0,'C:/Users/Clara Martins/Documents/Doutorado/Pierre Code/Bixi_poly/')
import config
from modelUtils import mini, maxi, proba_sum, proba_diff
import copy

class ServiceLevel(object):
    def __init__(self, env1,env2, mod1,mod2, beta1,beta2):
        """
        initialize information for computing service level
        :param env: environement information (Environment object)
        :param mod: model of class ModelStation
        :param arr_vs_dep: float between 0 and 1, how to valuedeparture vs arrival.  
        """
        self.env1 = env1
        self.env2 = env2
        self.mod1 = mod1
        self.mod2 = mod2
        self.mean1 = None
        self.mean2 = None
        self.var = None
        self.dict = {}
        pre = 1e-10

        # arr_vs_dep = min(1-pre,max(pre,arr_vs_dep))
        # self.arr = arr_vs_dep
        # self.dep = 1-arr_vs_dep
        
    def compute_mean_var(self, WT, data1,data2, predict):
        """
        compute and store estimated mean and variance
        :param WT: features to predict
        :param data: Data object with station information 
        :param predict: True: predict the mean, else use real data
        :return: None
        """
        # print(self.mod.reduce)
        # print(type(self.mod.reduce))
        try:
            self.dict['cols'] = self.mod1.reduce.preselect
        except AttributeError:
            self.dict['cols'] = self.mod1.models[0].reduce.preselect
        self.dict['arr_cols'] = data1.get_arr_cols(None)
        self.dict['stations'] = data1.get_stations_ids(None)
        self.dict['capacities'] = data1.get_stations_capacities(None).to_numpy().flatten()
        # self.dict['bike_type'] = data.env.
        # print("aqui")
        demand_ele = copy.deepcopy(pd.DataFrame(self.mod2.predict(x=WT), columns=self.dict['cols'])[
                self.dict['cols']])
        # print(demand_ele)
        for s in self.dict['stations']:
                x = copy.deepcopy((demand_ele['End date '+str(s)]+ demand_ele['Start date '+str(s)])*0.1)
                demand_ele['End date '+str(s)]= demand_ele['End date '+str(s)] + x
                demand_ele['Start date '+str(s)]= demand_ele['Start date '+str(s)] + x
        # print(demand_ele)
        # print(pd.DataFrame(self.mod1.predict(x=WT), columns=self.dict['cols'])[
        #         self.dict['cols']].to_numpy())
        # print(pd.DataFrame(self.mod1.predict(x=WT), columns=self.dict['cols'])[
        #         self.dict['cols']])
        if predict:
            self.mean1 = pd.DataFrame(self.mod1.predict(x=WT), columns=self.dict['cols'])[
                self.dict['cols']].to_numpy() #
            self.mean2 = demand_ele.to_numpy()
        else:
            print("error")
        #     if config.learning_var.__contains__('Heure'):
        #         dd = pd.merge(WT, data1.get_miniOD(), 'left',
        #                       on=['Mois', 'Jour', 'Annee', 'Heure'])
        #     else:
        #         dd = pd.merge(WT, data1.get_miniOD(), 'left',
        #                       on=['Mois', 'Jour', 'Annee', 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9',
        #                           'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19', 'h20', 'h21',
        #                           'h22', 'h23'])
        #     self.mean1 = dd[self.dict['cols']].to_numpy()
        self.var1 = pd.DataFrame(self.mod1.variance(x=WT), columns=self.dict['cols'])[
            self.dict['cols']].to_numpy()
        self.var2 = pd.DataFrame(self.mod2.variance(x=WT), columns=self.dict['cols'])[
            self.dict['cols']].to_numpy()
        self.var1[self.var1 == 0] = 0.01
        self.var2[self.var2 == 0] = 0.01
        self.stations_propor_reg = data1.get_stations_proportions_reg()

    def compute_service_gaussian(self, current_capacity=None):
        """
        compute the service level with a gaussian distribution
        :param current_capacity: current state of the network, if None compute the service for all capacities
        :return: service level : array if current_capacity set, else return a matrix with a service levels: each column 
            correspond to one station
        """
        cum_var = pd.DataFrame(np.cumsum(self.var, axis=0), columns=self.dict['cols'])
        cum_mean = pd.DataFrame(np.cumsum(self.mean, axis=0), columns=self.dict['cols'])
        m = pd.DataFrame(self.mean, columns=self.dict['cols'])
        arr = m[self.dict['arr_cols']].to_numpy()
        dep = m.drop(self.dict['arr_cols'], axis=1).to_numpy()
        for s in self.dict['stations']:
            cum_mean[str(s)] = cum_mean['End date ' + str(s)].to_numpy() - cum_mean[
                'Start date ' + str(s)].to_numpy()
            cum_var[str(s)] = cum_var['End date ' + str(s)].to_numpy() + cum_var[
                'Start date ' + str(s)].to_numpy()
        self.dict['cum_mean'] = cum_mean[list(map(str, self.dict['stations']))].to_numpy()
        self.dict['cum_var'] = cum_var[list(map(str, self.dict['stations']))].to_numpy()
        if current_capacity is None:
            service = np.zeros((np.max(self.dict['capacities'] + 1), dep.shape[1]))
            for c in range(np.max(self.dict['capacities']) + 1):
                cap = np.ones(dep.shape[1]) * c
                cum_mean = np.add(self.dict['cum_mean'], cap)
                proba_empty = norm.cdf(0, loc=cum_mean, scale=self.dict['cum_var'])
                proba_full = norm.sf(np.array(self.dict['capacities']), loc=cum_mean, scale=self.dict['cum_var'])
                service_loc = (dep * (1 - proba_empty)).sum(axis=0) / (np.sum(dep, axis=0) + 0.001)
                service_ret = (arr * (1 - proba_full)).sum(axis=0) / (np.sum(arr, axis=0) + 0.001)
                service[c] = 2 * mini(self.dep * service_loc, self.arr * service_ret)
                service[c, c > self.dict['capacities']] = 0
        else:
            cum_mean = np.add(self.dict['cum_mean'], current_capacity)
            proba_empty = norm.cdf(0, loc=cum_mean, scale=self.dict['cum_var'])
            proba_full = norm.sf(np.array(self.dict['capacities']), loc=cum_mean, scale=self.dict['cum_var'])
            service_loc = (dep * (1 - proba_empty)).sum(axis=0) / (np.sum(dep, axis=0) + 0.001)
            service_ret = (arr * (1 - proba_full)).sum(axis=0) / (np.sum(arr, axis=0) + 0.001)
            service = 2 * mini(self.dep * service_loc, self.arr * service_ret)
        return service

    def compute_service_level_from_proba_matrix(self, mat, available_bikes=None):
        """
        compute service level from the probability matrix, for any distribution (defined by the probability matrix)
        :param mat: matrix timewindow*n_stations*N, mat[t,s,k] is the proba that there are k departures (arrivals) at 
                time t at station s
        :param available_bikes: current network status, if None compute service for all statuses
        :return: service level : array if available_bikes set, else return a matrix with a service levels: each column 
            correspond to one station
        """
        res_mat = np.zeros(mat.shape)
        # compute the probability matrix of dep-arr for each station and each hour (cumulative)
        for t in range(mat.shape[0]):
            for i in range(mat.shape[1]):
                if t == 0:
                    res_mat[t, i, :] = mat[t, i, :]
                else:
                    p = proba_sum(res_mat[t - 1, i, :], mat[t, i, :])
                    p[res_mat.shape[2] - 1] = p[res_mat.shape[2] - 1:].sum()
                    res_mat[t, i, :] = p[:res_mat.shape[2]]
        p_dep_minus_arr = np.zeros((mat.shape[0], int(mat.shape[1] / 2), 2 * mat.shape[2] - 1))
        for s in range(int(mat.shape[1] / 2)):
            for t in range(mat.shape[0]):
                p_dep_minus_arr[t, s, :] = proba_diff(res_mat[t, 2 * s + 1, :], res_mat[t, 2 * s, :])
        # compute the probability of being superior or inferior or k (axis 2)
        proba_dep_inf = np.cumsum(p_dep_minus_arr, axis=2)
        proba_dep_sup = 1 - proba_dep_inf
        proba_dep_inf = proba_dep_inf - p_dep_minus_arr
        # get expected number of departure and arrivals
        m = pd.DataFrame(self.mean, columns=self.dict['cols'])
        arr = m[self.dict['arr_cols']].to_numpy()
        dep = m.drop(self.dict['arr_cols'], axis=1).to_numpy()
        if not (available_bikes is None):
            available_doks = self.dict['capacities'] - available_bikes
            available_doks = maxi(int(p_dep_minus_arr.shape[2] / 2) - available_doks, 0)
            available_bikes += int(p_dep_minus_arr.shape[2] / 2)
            available_bikes = mini(available_bikes, int(p_dep_minus_arr.shape[2]) - 1)
            available_bikes = available_bikes.astype(dtype=int)
            available_doks = available_doks.astype(dtype=int)
            # compute service level
            service_level_dep = (dep * proba_dep_inf[:, range(proba_dep_inf.shape[1]), available_bikes]).sum(
                axis=0) / dep.sum(axis=0)
            service_level_arr = (arr * proba_dep_sup[:, range(proba_dep_sup.shape[1]), available_doks]).sum(
                axis=0) / arr.sum(axis=0)
            service = 2 * mini(self.dep * service_level_dep, self.arr * service_level_arr)
        else:
            service = np.zeros((np.max(self.dict['capacities'] + 1), dep.shape[1]))
            for c in range(np.max(self.dict['capacities'])):
                available_bikes = mini(self.dict['capacities'], c)
                available_doks = self.dict['capacities'] - available_bikes
                available_doks = maxi(int(p_dep_minus_arr.shape[2] / 2) - available_doks, 0)
                available_bikes += int(p_dep_minus_arr.shape[2] / 2)
                available_bikes = mini(available_bikes, int(p_dep_minus_arr.shape[2]) - 1)
                available_bikes = available_bikes.astype(dtype=int)
                available_doks = available_doks.astype(dtype=int)
                # compute service level
                service_level_dep = (dep * proba_dep_inf[:, range(proba_dep_inf.shape[1]), available_bikes]).sum(
                    axis=0) / dep.sum(axis=0)
                service_level_arr = (arr * proba_dep_sup[:, range(proba_dep_sup.shape[1]), available_doks]).sum(
                    axis=0) / arr.sum(axis=0)
                service[c] = 2 * mini(self.dep * service_level_dep, self.arr * service_level_arr)
                service[c, c > self.dict['capacities']] = 0
        return service

    # def compute_service_level_Poisson(self, available_bikes=None):
    #     #print("Service Level")
    #     # lambdas = np.cumsum(self.mean,axis=0)
    #     cum_mean = pd.DataFrame(np.cumsum(self.mean, axis=0), columns=self.dict['cols']) #lista cumulativa por coluna 
        
    #     m = pd.DataFrame(self.mean, columns=self.dict['cols']) # viagem previstas para as proximas horas
    #     arr = m[self.dict['arr_cols']] #get the trips of the stations "End data 6000" 
    #     arr = arr.to_numpy()
    #     #print("passou")
    #     dep = m.drop(self.dict['arr_cols'], axis=1).to_numpy() #get the trips of the stations "Start data 6000" 
    #     # for s in self.dict['stations']:
    #     #     cum_mean[str(s)] = cum_mean['End date ' + str(s)].to_numpy() - cum_mean[
    #     #         'Start date ' + str(s)].to_numpy()
        
    #     cum_arr = cum_mean[self.dict['arr_cols']] #lista cumulativa End
    #     cum_dep = cum_mean.drop(self.dict['arr_cols'], axis=1) #lista cumulativa start
    #     # self.dict['cum_mean'] = cum_mean[list(map(str, self.dict['stations']))].to_numpy()
        
    #     if available_bikes is None:
        
    #         service = np.zeros((np.max(self.dict['capacities'] + 1), dep.shape[1])) #a maior capacidade+1, estações
        
    #         for c in range(np.max(self.dict['capacities']) + 1):
    #             cap = np.ones(dep.shape[1]) * c
    #             loc = cap # array com quantidade de estações e todos os valores iguais a c
    #             # cum_mean = np.add(self.dict['cum_mean'], cap)
    #             # print("loc")
    #             # print(loc)
    #             proba_empty = skellam.cdf(0, mu1=cum_arr, mu2=cum_dep, loc=loc)
    #             proba_full = skellam.sf(np.array(self.dict['capacities']), mu1=cum_arr, mu2=cum_dep, loc=loc)
    #             #print("proba_empty")
    #             #print(np.size(proba_empty))
    #             service_loc = (dep * (1 - proba_empty)).sum(axis=0) / (np.sum(dep, axis=0) + 0.001) #Service level rent
    #             service_ret = (arr * (1 - proba_full)).sum(axis=0) / (np.sum(arr, axis=0) + 0.001) #Service level return
                
    #             service[c] = 2 * mini(self.dep * service_loc, self.arr * service_ret)
    #             service[c, c > self.dict['capacities']] = 0
    #     else:
    #         loc = np.ones(cum_arr.shape) * available_bikes
    #         proba_empty = skellam.cdf(0, mu1=cum_arr, mu2=cum_dep, loc=loc)
    #         proba_full = skellam.sf(np.array(self.dict['capacities']), mu1=cum_arr, mu2=cum_dep, loc=loc)
    #         service_loc = (dep * (1 - proba_empty)).sum(axis=0) / (np.sum(dep, axis=0) + 0.001)
    #         service_ret = (arr * (1 - proba_full)).sum(axis=0) / (np.sum(arr, axis=0) + 0.001)
    #         service = 2 * mini(self.dep * service_loc, self.arr * service_ret)
    #     return service

    def compute_service_level_Poisson3(self, available_bikes=None):
        #print("Service Level")
            # lambdas = np.cumsum(self.mean,axis=0)
            
            # df.to_csv("cap_total.csv")
            # df_reg = pd.DataFrame([])
            # df_ele = pd.DataFrame([])
        cum_mean1 = pd.DataFrame(np.cumsum(self.mean1, axis=0), columns=self.dict['cols']) #lista cumulativa por coluna 
        cum_mean2 = pd.DataFrame(np.cumsum(self.mean2, axis=0), columns=self.dict['cols'])

        m1 = pd.DataFrame(self.mean1, columns=self.dict['cols']) # viagem previstas para as proximas horas
        m2 = pd.DataFrame(self.mean2, columns=self.dict['cols'])
        arr1 = m1[self.dict['arr_cols']].to_numpy() #get the trips of the stations "End data 6000" 
        arr2 = m2[self.dict['arr_cols']].to_numpy()
        dep1 = m1.drop(self.dict['arr_cols'], axis=1).to_numpy() #get the trips of the stations "Start data 6000" 
        dep2 = m2.drop(self.dict['arr_cols'], axis=1).to_numpy()
        
        cum_arr1 = cum_mean1[self.dict['arr_cols']] #lista cumulativa End
        cum_dep1 = cum_mean1.drop(self.dict['arr_cols'], axis=1) #lista cumulativa start
        cum_arr2 = cum_mean2[self.dict['arr_cols']] #lista cumulativa End
        cum_dep2 = cum_mean2.drop(self.dict['arr_cols'], axis=1) #lista cumulativa start
        # self.dict['cum_mean'] = cum_mean[list(map(str, self.dict['stations']))].to_numpy()
        
        # if available_bikes is None:
        
        service_reg = np.zeros((np.max(self.dict['capacities'] + 1), dep1.shape[1])) #a maior capacidade+1, estações
        service_ele = np.zeros((np.max(self.dict['capacities'] + 1), dep1.shape[1]))
        best_service_joint = np.zeros( dep1.shape[1])
        best_service_reg = np.zeros((np.max(self.dict['capacities'] + 1), dep1.shape[1])) 
        best_service_ele = np.zeros((np.max(self.dict['capacities'] + 1), dep1.shape[1])) 
        cap_reg = np.zeros( dep1.shape[1])
        cap_ele = np.zeros( dep1.shape[1])
        
        for y in range(1,np.max(self.dict['capacities']) + 1):
            # service_reg = np.zeros((np.max(self.dict['capacities'] + 1), dep1.shape[1])) #a maior capacidade+1, estações
            # service_ele = np.zeros((np.max(self.dict['capacities'] + 1), dep1.shape[1]))
            for c in range(np.max(self.dict['capacities']) + 1):
                
                cap = np.ones(dep1.shape[1]) * c
                loc = cap # array com quantidade de estações e todos os valores iguais a c
                
                # cap = np.ones(dep1.shape[1]) * (np.max(self.dict['capacities'])-c)
                # loc2 = cap #

                capacity_reg = np.array(self.dict['capacities'])
                capacity_reg[capacity_reg > y] = y

                capacity_ele = np.array(self.dict['capacities']) - capacity_reg
                # print(capacity_ele)
                proba_empty1 = skellam.cdf(0, mu1=cum_arr1, mu2=cum_dep1, loc=loc)
                proba_empty2 = skellam.cdf(0, mu1=cum_arr2, mu2=cum_dep2, loc=loc)

                proba_full1 = skellam.sf(capacity_reg, mu1=cum_arr1, mu2=cum_dep1, loc=loc)
                proba_full2 = skellam.sf(capacity_ele, mu1=cum_arr2, mu2=cum_dep2, loc=loc)
                
                service_loc_re = (dep1 * (1 - proba_empty1)).sum(axis=0) / ((np.sum(dep1, axis=0) + 0.001)+(np.sum(dep2, axis=0) + 0.001)) #Service level rent
                service_ret_re = (arr1 * (1 - proba_full1)).sum(axis=0) / ((np.sum(arr1, axis=0) + 0.001)+(np.sum(arr2, axis=0) + 0.001)) #Service level return
                service_loc_el = (dep2 * (1 - proba_empty2)).sum(axis=0) / ((np.sum(dep1, axis=0) + 0.001)+(np.sum(dep2, axis=0) + 0.001)) #Service level rent
                service_ret_el = (arr2 * (1 - proba_full2)).sum(axis=0) / ((np.sum(arr1, axis=0) + 0.001)+(np.sum(arr2, axis=0) + 0.001)) #Service level return
                            
                # service_reg[c] = 2 * mini( service_loc_re,  service_ret_re)
                # pro_dep = (np.sum(dep1, axis=0) + 0.001)/((np.sum(dep1, axis=0) + 0.001)+(np.sum(dep2, axis=0) + 0.001))
                # pro_arr = (np.sum(arr1, axis=0) + 0.001)/((np.sum(arr1, axis=0) + 0.001)+(np.sum(arr2, axis=0) + 0.001))
                
                service_reg[c] = (service_loc_re + service_ret_re)/2#/((np.sum(dep1, axis=0) + 0.001)+(np.sum(arr1, axis=0) + 0.001))
                service_reg[c, c > capacity_reg] = 0
                
                # service_ele[c] = 2 * mini( service_loc_el,  service_ret_el)
                # pro_dep = (np.sum(dep2, axis=0) + 0.001)/((np.sum(dep1, axis=0) + 0.001)+(np.sum(dep2, axis=0) + 0.001))
                # pro_arr = (np.sum(arr2, axis=0) + 0.001)/((np.sum(arr1, axis=0) + 0.001)+(np.sum(arr2, axis=0) + 0.001))
                
                service_ele[c] = (service_loc_el+ service_ret_el)/2#/((np.sum(dep2, axis=0) + 0.001)+(np.sum(arr2, axis=0) + 0.001))
                service_ele[c, c > capacity_ele] = 0    

       
            service_joint = service_reg.max(axis=0) + service_ele.max(axis=0)
            
            # service_joint = np.array([i[i!=0].min() for i in service_reg.T]) + np.array([i[i!=0].min() for i in service_ele.T])
            # print(service_joint)
            # service_joint = service_reg.sum(0)/(service_reg!=0).sum(0).astype(float) + service_ele.sum(0)/(service_ele!=0).sum(0).astype(float)

            # service_joint = mini(service_reg.max(axis=0) , service_ele.max(axis=0))

            for j in range(service_joint.shape[0]):
                if service_joint[j] >= best_service_joint[j]:
                    best_service_joint [j] = copy.deepcopy(service_joint[j])
                    best_service_reg[:,j] = copy.deepcopy(service_reg[:,j])
                    best_service_ele[:,j] = copy.deepcopy(service_ele[:,j]) 
                    cap_reg[j] = copy.deepcopy(capacity_reg[j])
                    cap_ele[j] = copy.deepcopy(capacity_ele[j]) 

        # print("2")
        # print(best_service_reg)
        # print(best_service_ele)
        return best_service_reg,best_service_ele

      

    def compute_service_level_Poisson2(self,model, available_bikes=None):
        #print("Service Level")
        # lambdas = np.cumsum(self.mean,axis=0)
        cum_mean1 = pd.DataFrame(np.cumsum(self.mean1, axis=0), columns=self.dict['cols']) #lista cumulativa por coluna 
        cum_mean2 = pd.DataFrame(np.cumsum(self.mean2, axis=0), columns=self.dict['cols'])

        m1 = pd.DataFrame(self.mean1, columns=self.dict['cols']) # viagem previstas para as proximas horas
        m2 = pd.DataFrame(self.mean2, columns=self.dict['cols'])

        arr1 = m1[self.dict['arr_cols']].to_numpy() #get the trips of the stations "End data 6000" 
        arr2 = m2[self.dict['arr_cols']].to_numpy()
        dep1 = m1.drop(self.dict['arr_cols'], axis=1).to_numpy() #get the trips of the stations "Start data 6000" 
        dep2 = m2.drop(self.dict['arr_cols'], axis=1).to_numpy()
        
        cum_arr1 = cum_mean1[self.dict['arr_cols']] #lista cumulativa End
        cum_dep1 = cum_mean1.drop(self.dict['arr_cols'], axis=1) #lista cumulativa start
        cum_arr2 = cum_mean2[self.dict['arr_cols']] #lista cumulativa End
        cum_dep2 = cum_mean2.drop(self.dict['arr_cols'], axis=1) #lista cumulativa start
        # self.dict['cum_mean'] = cum_mean[list(map(str, self.dict['stations']))].to_numpy()
        
        # if available_bikes is None:
        
        service_reg = np.zeros((np.max(self.dict['capacities'] + 1), dep1.shape[1])) #a maior capacidade+1, estações
        service_ele = np.zeros((np.max(self.dict['capacities'] + 1), dep1.shape[1]))
        best_service_joint = np.zeros( dep1.shape[1])
        best_service_reg = np.zeros((np.max(self.dict['capacities'] + 1), dep1.shape[1])) 
        best_service_ele = np.zeros((np.max(self.dict['capacities'] + 1), dep1.shape[1])) 
        cap_reg = np.zeros( dep1.shape[1])
        cap_ele = np.zeros( dep1.shape[1])
        for y in range(1,np.max(self.dict['capacities']) + 1):
            # service_reg = np.zeros((np.max(self.dict['capacities'] + 1), dep1.shape[1])) #a maior capacidade+1, estações
            # service_ele = np.zeros((np.max(self.dict['capacities'] + 1), dep1.shape[1]))
            for c in range(np.max(self.dict['capacities']) + 1):
                
                cap = np.ones(dep1.shape[1]) * c
                loc = cap # array com quantidade de estações e todos os valores iguais a c
                
                # cap = np.ones(dep1.shape[1]) * (np.max(self.dict['capacities'])-c)
                # loc2 = cap #
                
                capacity_reg = np.array(self.dict['capacities'])
                capacity_reg[capacity_reg > y] = y

                capacity_ele = np.array(self.dict['capacities']) - capacity_reg
                # print(capacity_ele)
                proba_empty1 = skellam.cdf(0, mu1=cum_arr1, mu2=cum_dep1, loc=loc)
                proba_empty2 = skellam.cdf(0, mu1=cum_arr2, mu2=cum_dep2, loc=loc)

                proba_full1 = skellam.sf(capacity_reg, mu1=cum_arr1, mu2=cum_dep1, loc=loc)
                proba_full2 = skellam.sf(capacity_ele, mu1=cum_arr2, mu2=cum_dep2, loc=loc)
                
                service_loc_re = (dep1 * (1 - proba_empty1)).sum(axis=0)# / (np.sum(dep1, axis=0) + 0.001) #Service level rent
                service_ret_re = (arr1 * (1 - proba_full1)).sum(axis=0) #/ (np.sum(arr1, axis=0) + 0.001) #Service level return
                service_loc_el = (dep2 * (1 - proba_empty2)).sum(axis=0) #/ (np.sum(dep2, axis=0) + 0.001) #Service level rent
                service_ret_el = (arr2 * (1 - proba_full2)).sum(axis=0) #/ (np.sum(arr2, axis=0) + 0.001) #Service level return
                            
                # service_reg[c] = 2 * mini( service_loc_re,  service_ret_re)
                service_reg[c] = ( service_loc_re+ service_ret_re)/((np.sum(dep1, axis=0) + 0.001)+(np.sum(arr1, axis=0) + 0.001))
                service_reg[c, c > capacity_reg] = 0
                
                # service_ele[c] = 2 * mini( service_loc_el,  service_ret_el)
                service_ele[c] = ( service_loc_el+ service_ret_el)/((np.sum(dep2, axis=0) + 0.001)+(np.sum(arr2, axis=0) + 0.001))
                service_ele[c, c > capacity_ele] = 0    

            # Mudar aqui
            if model ==1:
                service_joint = service_reg.sum(0)/(service_reg!=0).sum(0).astype(float) + service_ele.sum(0)/(service_ele!=0).sum(0).astype(float)
            elif model ==2:
                service_joint = service_reg.max(axis=0) + service_ele.max(axis=0)
            else:
                service_joint = service_reg.min(axis=0) + service_ele.min(axis=0)

            for j in range(service_joint.shape[0]):
                if service_joint[j] >= best_service_joint[j]:
                    best_service_joint [j] = service_joint[j]
                    best_service_reg[:,j] = service_reg[:,j]
                    best_service_ele[:,j] = service_ele[:,j] 
                    cap_reg[j] = capacity_reg[j]
                    cap_ele[j] = capacity_ele[j] 
        
        return best_service_reg,best_service_ele

    def compute_service_level_Poisson(self, available_bikes=None):
        #print("Service Level")
        # lambdas = np.cumsum(self.mean,axis=0)
        
        # df.to_csv("cap_total.csv")
        # df_reg = pd.DataFrame([])
        # df_ele = pd.DataFrame([])
        cum_mean1 = pd.DataFrame(np.cumsum(self.mean1, axis=0), columns=self.dict['cols']) #lista cumulativa por coluna 
        cum_mean2 = pd.DataFrame(np.cumsum(self.mean2, axis=0), columns=self.dict['cols'])

        m1 = pd.DataFrame(self.mean1, columns=self.dict['cols']) # viagem previstas para as proximas horas
        m2 = pd.DataFrame(self.mean2, columns=self.dict['cols'])
        arr1 = m1[self.dict['arr_cols']].to_numpy() #get the trips of the stations "End data 6000" 
        arr2 = m2[self.dict['arr_cols']].to_numpy()
        dep1 = m1.drop(self.dict['arr_cols'], axis=1).to_numpy() #get the trips of the stations "Start data 6000" 
        dep2 = m2.drop(self.dict['arr_cols'], axis=1).to_numpy()
        
        cum_arr1 = cum_mean1[self.dict['arr_cols']] #lista cumulativa End
        cum_dep1 = cum_mean1.drop(self.dict['arr_cols'], axis=1) #lista cumulativa start
        cum_arr2 = cum_mean2[self.dict['arr_cols']] #lista cumulativa End
        cum_dep2 = cum_mean2.drop(self.dict['arr_cols'], axis=1) #lista cumulativa start
        # self.dict['cum_mean'] = cum_mean[list(map(str, self.dict['stations']))].to_numpy()
        
        # if available_bikes is None:
        
        service_reg = np.zeros((np.max(self.dict['capacities'] + 1), dep1.shape[1])) #a maior capacidade+1, estações
        service_ele = np.zeros((np.max(self.dict['capacities'] + 1), dep1.shape[1]))
        best_service_joint = np.zeros( dep1.shape[1])
        best_service_reg = np.zeros((np.max(self.dict['capacities'] + 1), dep1.shape[1])) 
        best_service_ele = np.zeros((np.max(self.dict['capacities'] + 1), dep1.shape[1])) 
        cap_reg = np.zeros( dep1.shape[1])
        cap_ele = np.zeros( dep1.shape[1])
        for y in range(1,np.max(self.dict['capacities']) + 1):
            # service_reg = np.zeros((np.max(self.dict['capacities'] + 1), dep1.shape[1])) #a maior capacidade+1, estações
            # service_ele = np.zeros((np.max(self.dict['capacities'] + 1), dep1.shape[1]))
            for c in range(np.max(self.dict['capacities']) + 1):
                
                cap = np.ones(dep1.shape[1]) * c
                loc = cap # array com quantidade de estações e todos os valores iguais a c
                
                # cap = np.ones(dep1.shape[1]) * (np.max(self.dict['capacities'])-c)
                # loc2 = cap #
                
                capacity_reg = np.array(self.dict['capacities'])
                capacity_reg[capacity_reg > y] = y

                capacity_ele = np.array(self.dict['capacities']) - capacity_reg
                # print(capacity_ele)
                proba_empty1 = skellam.cdf(0, mu1=cum_arr1, mu2=cum_dep1, loc=loc)
                proba_empty2 = skellam.cdf(0, mu1=cum_arr2, mu2=cum_dep2, loc=loc)

                proba_full1 = skellam.sf(capacity_reg, mu1=cum_arr1, mu2=cum_dep1, loc=loc)
                proba_full2 = skellam.sf(capacity_ele, mu1=cum_arr2, mu2=cum_dep2, loc=loc)
                
                service_loc_re = (dep1 * (1 - proba_empty1)).sum(axis=0)/ (np.sum(dep1, axis=0) + 0.001) #Service level rent
                service_ret_re = (arr1 * (1 - proba_full1)).sum(axis=0)/ (np.sum(arr1, axis=0) + 0.001) #Service level return
                service_loc_el = (dep2 * (1 - proba_empty2)).sum(axis=0) / (np.sum(dep2, axis=0) + 0.001) #Service level rent
                service_ret_el = (arr2 * (1 - proba_full2)).sum(axis=0) / (np.sum(arr2, axis=0) + 0.001) #Service level return
                            
                service_reg[c] = 2 * mini( service_loc_re,  service_ret_re)
                # service_reg[c] = ( service_loc_re+ service_ret_re)/((np.sum(dep1, axis=0) + 0.001)+(np.sum(arr1, axis=0) + 0.001))
                service_reg[c, c > capacity_reg] = 0
                
                service_ele[c] = 2 * mini( service_loc_el,  service_ret_el)
                # service_ele[c] = ( service_loc_el+ service_ret_el)/((np.sum(dep2, axis=0) + 0.001)+(np.sum(arr2, axis=0) + 0.001))
                service_ele[c, c > capacity_ele] = 0    

            if model ==1:
                service_joint = service_reg.max(axis=0) + service_ele.max(axis=0)
            elif model ==2:
                service_joint = service_reg.sum(0)/(service_reg!=0).sum(0).astype(float) + service_ele.sum(0)/(service_ele!=0).sum(0).astype(float)            
            else:
                service_joint = service_reg.min(axis=0) + service_ele.min(axis=0)

            for j in range(service_joint.shape[0]):
                if service_joint[j] >= best_service_joint[j]:
                    best_service_joint [j] = service_joint[j]
                    best_service_reg[:,j] = service_reg[:,j]
                    best_service_ele[:,j] = service_ele[:,j] 
                    cap_reg[j] = capacity_reg[j]
                    cap_ele[j] = capacity_ele[j] 
        
        return best_service_reg,best_service_ele
      


    def compute_proba_matrix(self, distrib='NB'):
        """
        computes the probability of number of departure and arrivals matrix for a given distribution 
        :param distrib: the distribution of proba, default NB, available : Poisson (P) and Zero Inflated(ZI)
        :return: the matrix (timewindow*(2n_stations)*N)
        """
        self.N = 80
        mat = np.zeros((self.mean.shape[0], self.mean.shape[1], self.N))
        if distrib == 'NB':
            p = self.mean / self.var
            p = mini(p, 0.999)
            r = maxi(1, (self.mean * p / (1 - p)))
            r = mini(150, r)
            n = np.array(list(map(int, r.flatten()))).reshape(r.shape)
            n += 1
            for k in range(self.N):
                mat[:, :, k] = nbinom.pmf(k, n=n, p=p)
                # a = comb(n + k - 1, k)
                # b = (p ** n) * ((1 - p) ** k)
                # mat[:, :, k] = a * b
        elif distrib == 'P':
            p = self.mean
            for k in range(self.N):
                mat[:, :, k] = poisson.pmf(k, p)
            sum = maxi(1 - mat.sum(axis=2), 0)
            mat[:, :, -1] += sum
        elif distrib == 'ZI':
            l = (self.var + self.mean ** 2) / self.mean - 1
            l[np.isnan(l)] = 0
            l[l < 0] = 0
            # l[self.mean == 0] = 0
            l = maxi(l, self.mean)
            # l = maxi(l, self.var)
            l = mini(l / self.mean, 50) * self.mean

            pi = 1 - self.mean / l
            # pi[l == 0] = 1
            # pi[l == 1000] = 1
            # pi[pi < 0] = 0
            for k in range(self.N):
                mat[:, :, k] = poisson.pmf(k, l)
                # np.exp(-l)*(l**k)/factorial(l)
                mat[:, :, k] *= 1 - pi
            mat[:, :, 0] += pi
            sum = maxi(1 - mat.sum(axis=2), 0)
            mat[:, :, -1] += sum

        # print('mat sum',(mat.sum(axis=2)-1).sum())
        assert (mat.sum(axis=2) - 1).sum() < 1e-6
        return mat

    def compute_service_level(self, model =1,current_capacity=None):
        """
        compute the service level, NB distribution supposed
        :param current_capacity: current network status, if None compute service for all statuses
        :return: service level : array if available_bikes set, else return a matrix with a service levels: each column 
            correspond to one station
        """
        # if distribution == 'P':
        # if model == 1: 
        #     print("Function Max")
        #     # service = self.compute_service_level_Poisson()
        # elif model==2:
        #     print("Function Avg")
        #     # service = self.compute_service_level_Poisson2()
        # elif model ==3:
        #     print("Function Min")
        #     # service = self.compute_service_level_Poisson3()
        # else:
        #     print("error")
        #     return
        service  = self.compute_service_level_Poisson2(model=model)
        # service  = self.compute_service_level_Poisson2_plus(model=model,plus=0.1)
        
        # if model == 1: 
        #     print("model1")
        #     service = self.compute_service_level_Poisson()
        # elif model==2:
        #     print("model2")
        #     service = self.compute_service_level_Poisson2()
        # elif model ==3:
        #     print("model3")
        #     service = self.compute_service_level_Poisson3()
        # # else:
        #     mat = self.compute_proba_matrix(distribution)
        #     service = self.compute_service_level_from_proba_matrix(mat, current_capacity)
        return service


if __name__ == '__main__':
    from model_station.ModelStations import ModelStations
    from preprocessing.Environment import Environment
    from preprocessing.Data import Data
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    ud = Environment('Bixi', 'train')
    d=Data(ud)
    mod= ModelStations(ud,'svd','gbt',**{'var':True})
    mod.train(d)
    mod.save()
    mod.load()
    i=43
    print(d.get_stations_col()[i], d.get_miniOD([])[d.get_stations_col()[i]].mean())
    sl = ServiceLevel(Environment('Bixi', 'train'), mod, 0.5)
    cmap = cm.binary
    s=0
    dd=[]
    for k in [10]:
        WT = mod.get_factors(d).iloc[:k,:]
        sl.compute_mean_var(WT,d,True)
        cap = sl.dict['capacities'][i]
        service = sl.compute_service_level_Poisson()
        np.savetxt('test.csv',service,)
        plt.plot(service[:,i][:cap], c=cmap(k/100 ))
        dd.append(np.log(((service[:,i]-s)**2).sum()))
        s=service[:,i]
        # print(service[5])
    plt.show()
    plt.plot(range(2,100,2),dd)
    plt.show()