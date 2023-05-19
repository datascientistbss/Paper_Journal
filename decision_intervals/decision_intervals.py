from ServiceLevel import ServiceLevel
from modelUtils import *
from model_station.ModelStations import ModelStations
from preprocessing.Data import Data
from preprocessing.Environment import Environment
import operator
import copy
from decimal import Decimal, ROUND_HALF_UP,ROUND_HALF_DOWN
import datetime
import random
import numpy as np

class DecisionIntervals(object):
    """
    class for computing the decision intervals, uses the Service level class
    """
    def __init__(self, env1,env2, mod1, mod2, beta1, beta2):
        """
        :param env: environment
        :param mod: 
        :param arr_vs_dep: alpha value (value of arrivals vs departures)
        :param beta: strength of intervals
        """
        #Mudar
        #old period 2018
        # self.hours = [0, 9, 11, 15, 20] # 2018
        # self.length = [9, 2, 4, 5, 4] # 2018
        #New periods
        self.hours = [6, 9, 11, 16, 19, 22] #2019
        self.periods = np.size(self.hours)
        # self.year = 2019
        self.SL = ServiceLevel(env1,env2, mod1,mod2, beta1, beta2)
        self.param_beta1 = beta1
        self.param_beta2 = beta2

    def compute_decision_intervals(self, WT, data1,data2,model, predict=True, **kwargs):
        """
        computes decision intervals for WT (Weather and temporal features) data
        :param data: the learning data (class Data)
        :param predict: if True use predictions, if False use real data
        :return: (min, target, max), each element being a array containing one value per station
        """
        # hparam = {
        #     'distrib': 'P'
        # }
        # hparam.update(kwargs)
        
        self.SL.compute_mean_var(WT, data1,data2, predict) #predicting values
        service1, service2 = self.SL.compute_service_level(model=model) 
        
        #############################################        
        best_inv1 = np.argmax(service1, axis=0)#SL max
        best_serlev1 = service1[best_inv1, range(service1.shape[1])]
        service1[service1 == 0] = 2
        
        worst_inv1 = np.argmin(service1, axis=0) #SL min
        worst_serlev1 = service1[worst_inv1, range(service1.shape[1])]
        worst_serlev1[worst_serlev1 == 2] = 0
        
        service1[service1 == 2] = 0
        service_min_to_assure1 = worst_serlev1 + (best_serlev1 - worst_serlev1) * self.param_beta1
        
        b1 = service1 >= service_min_to_assure1
        b21 = np.cumsum(b1, axis=0)
        
        min_inv1 = np.argmax(b1, axis=0)
        max_inv1 = np.argmax(b21, axis=0)
        ###########################################3
        
        best_inv2 = np.argmax(service2, axis=0)#SL max
        best_serlev2 = service2[best_inv2, range(service2.shape[1])]
        service2[service2 == 0] = 2
        
        worst_inv2 = np.argmin(service2, axis=0) #SL min
        worst_serlev2 = service2[worst_inv2, range(service2.shape[1])]
        worst_serlev2[worst_serlev2 == 2] = 0
        
        service2[service2 == 2] = 0
        service_min_to_assure2 = worst_serlev2 + (best_serlev2 - worst_serlev2) * self.param_beta2
        
        b2 = service2 >= service_min_to_assure2
        b22 = np.cumsum(b2, axis=0)
        
        min_inv2 = np.argmax(b2, axis=0)
        max_inv2 = np.argmax(b22, axis=0)


        return min_inv1, best_inv1, max_inv1, min_inv2, best_inv2, max_inv2

    # def compute_decision_intervals2(self, WT, data1,data2, predict=True, **kwargs):
    #     """
    #     computes decision intervals for WT (Weather and temporal features) data
    #     :param data: the learning data (class Data)
    #     :param predict: if True use predictions, if False use real data
    #     :return: (min, target, max), each element being a array containing one value per station
    #     """
    #     # hparam = {
    #     #     'distrib': 'P'
    #     # }
    #     # hparam.update(kwargs)
    #     print("model2")
    #     self.SL.compute_mean_var(WT, data1,data2, predict) #predicting values
    #     service1, service2 = self.SL.compute_service_level(model =2) 
        
    #     #############################################        
    #     best_inv1 = np.argmax(service1, axis=0)#SL max
    #     best_serlev1 = service1[best_inv1, range(service1.shape[1])]
    #     service1[service1 == 0] = 2
        
    #     worst_inv1 = np.argmin(service1, axis=0) #SL min
    #     worst_serlev1 = service1[worst_inv1, range(service1.shape[1])]
    #     worst_serlev1[worst_serlev1 == 2] = 0
        
    #     service1[service1 == 2] = 0
    #     service_min_to_assure1 = worst_serlev1 + (best_serlev1 - worst_serlev1) * self.param_beta1
        
    #     b1 = service1 >= service_min_to_assure1
    #     b21 = np.cumsum(b1, axis=0)
        
    #     min_inv1 = np.argmax(b1, axis=0)
    #     max_inv1 = np.argmax(b21, axis=0)
    #     ###########################################3
        
    #     best_inv2 = np.argmax(service2, axis=0)#SL max
    #     best_serlev2 = service2[best_inv2, range(service2.shape[1])]
    #     service2[service2 == 0] = 2
        
    #     worst_inv2 = np.argmin(service2, axis=0) #SL min
    #     worst_serlev2 = service2[worst_inv2, range(service2.shape[1])]
    #     worst_serlev2[worst_serlev2 == 2] = 0
        
    #     service2[service2 == 2] = 0
    #     service_min_to_assure2 = worst_serlev2 + (best_serlev2 - worst_serlev2) * self.param_beta2
        
    #     b2 = service2 >= service_min_to_assure2
    #     b22 = np.cumsum(b2, axis=0)
        
    #     min_inv2 = np.argmax(b2, axis=0)
    #     max_inv2 = np.argmax(b22, axis=0)


    #     return min_inv1, best_inv1, max_inv1, min_inv2, best_inv2, max_inv2


    # def compute_decision_intervals3(self, WT, data1,data2, predict=True, **kwargs):
    #     """
    #     computes decision intervals for WT (Weather and temporal features) data
    #     :param data: the learning data (class Data)
    #     :param predict: if True use predictions, if False use real data
    #     :return: (min, target, max), each element being a array containing one value per station
    #     """
    #     # hparam = {
    #     #     'distrib': 'P'
    #     # }
    #     # hparam.update(kwargs)
    #     print("model3")
    #     self.SL.compute_mean_var(WT, data1,data2, predict) #predicting values
    #     service1, service2 = self.SL.compute_service_level3() 
        
    #     #############################################        
    #     best_inv1 = np.argmax(service1, axis=0)#SL max
    #     best_serlev1 = service1[best_inv1, range(service1.shape[1])]
    #     service1[service1 == 0] = 2
        
    #     worst_inv1 = np.argmin(service1, axis=0) #SL min
    #     worst_serlev1 = service1[worst_inv1, range(service1.shape[1])]
    #     worst_serlev1[worst_serlev1 == 2] = 0
        
    #     service1[service1 == 2] = 0
    #     service_min_to_assure1 = worst_serlev1 + (best_serlev1 - worst_serlev1) * self.param_beta1
        
    #     b1 = service1 >= service_min_to_assure1
    #     b21 = np.cumsum(b1, axis=0)
        
    #     min_inv1 = np.argmax(b1, axis=0)
    #     max_inv1 = np.argmax(b21, axis=0)
    #     ###########################################3
        
    #     best_inv2 = np.argmax(service2, axis=0)#SL max
    #     best_serlev2 = service2[best_inv2, range(service2.shape[1])]
    #     service2[service2 == 0] = 2
        
    #     worst_inv2 = np.argmin(service2, axis=0) #SL min
    #     worst_serlev2 = service2[worst_inv2, range(service2.shape[1])]
    #     worst_serlev2[worst_serlev2 == 2] = 0
        
    #     service2[service2 == 2] = 0
    #     service_min_to_assure2 = worst_serlev2 + (best_serlev2 - worst_serlev2) * self.param_beta2
        
    #     b2 = service2 >= service_min_to_assure2
    #     b22 = np.cumsum(b2, axis=0)
        
    #     min_inv2 = np.argmax(b2, axis=0)
    #     max_inv2 = np.argmax(b22, axis=0)


    #     return min_inv1, best_inv1, max_inv1, min_inv2, best_inv2, max_inv2

    def compute_min_max_data(self, WT, data1,data2,ext, predict=True, save=True, **kwargs):
        """
        compute the min/max/target intervals starting from the first monday in WT and saves it to file 
        :param WT: features to predict on 
        :param data: the enviroment data (Data object) for station information
        :param predict: predict: if True use predictions, if False use real data
        :return: intervals data frame
        """
        # print(WT)
        # print(WT['Jour'])
        # hparam.update(kwargs)
        tw =5
        cond = True
        i0 = 0
        while cond:
            cond = not ((WT['wday'].to_numpy()[i0] == 6) and (WT['Heure'].to_numpy()[i0] == 0)) #meia noite do domingo
            i0 += 1
        i0 -= 1 # quanto horas ate segunda de manha

        cols_min = []
        cols_max = []
        cols_target = []
        cols = []
        for i in range(1, 8):
            for t in range(1, self.periods+1): #Mudar
                cols_min.append('dayOfWeek' + str(i) + '.period' + str(t) + 'Min')
                cols_target.append('dayOfWeek' + str(i) + '.period' + str(t) + 'Target')
                cols_max.append('dayOfWeek' + str(i) + '.period' + str(t) + 'Max')
                cols.append('dayOfWeek' + str(i) + '.period' + str(t) + 'Min')
                cols.append('dayOfWeek' + str(i) + '.period' + str(t) + 'Target')
                cols.append('dayOfWeek' + str(i) + '.period' + str(t) + 'Max')
        

        inter_min_re = pd.DataFrame(np.zeros((data1.get_stations_capacities(None).shape[0], 7 * self.periods)), #Mudar
                                 index=data1.get_stations_ids(None), columns=cols_min)
        inter_max_re = pd.DataFrame(np.zeros((data1.get_stations_capacities(None).shape[0], 7 * self.periods)), # Criando um dataFrame linhas sao as estações e colunas cols_min
                                 index=data1.get_stations_ids(None), columns=cols_max)
        inter_target_re = pd.DataFrame(np.zeros((data1.get_stations_capacities(None).shape[0], 7 * self.periods)),
                                    index=data1.get_stations_ids(None), columns=cols_target)
        inter_min_el = pd.DataFrame(np.zeros((data2.get_stations_capacities(None).shape[0], 7 * self.periods)), #Mudar
                                 index=data2.get_stations_ids(None), columns=cols_min)
        inter_max_el = pd.DataFrame(np.zeros((data2.get_stations_capacities(None).shape[0], 7 * self.periods)), # Criando um dataFrame linhas sao as estações e colunas cols_min
                                 index=data2.get_stations_ids(None), columns=cols_max)
        inter_target_el = pd.DataFrame(np.zeros((data2.get_stations_capacities(None).shape[0], 7 * self.periods)),
                                    index=data2.get_stations_ids(None), columns=cols_target)
        #Aqui
        if ext=='model1':
            model=1
        elif ext=='model2':
            model=2
        elif ext=='model3':
            model=3
        else:
            print("Error")
            return

        for d in range(7): #dia da semana 
            for h in range(len(self.hours)): #perido
                print("day "+str(d))
                print("period "+str(h))
                print(i0 + 24 * d + self.hours[h] + tw, end='\r')
                # print(WT.iloc[i0:])
                assert (WT['wday'].iloc[i0 + 24 * d + self.hours[h]] == (d - 1) % 7)

                # if ext=='model1':
                #     m1, t1, M1, m2, t2, M2 = self.compute_decision_intervals(WT=
                #         WT.iloc[i0 + 24 * d + self.hours[h]:i0 + 24 * d + self.hours[h] + tw, :], data1=data1,data2=data2,model=model,
                #         predict=predict, **kwargs)
                # elif ext=='model2':
                #     m1, t1, M1, m2, t2, M2 = self.compute_decision_intervals2(WT=
                #         WT.iloc[i0 + 24 * d + self.hours[h]:i0 + 24 * d + self.hours[h] + tw, :], data1=data1,data2=data2,model=model,
                #         predict=predict, **kwargs)
                # elif ext=='model3':
                m1, t1, M1, m2, t2, M2 = self.compute_decision_intervals(WT=
                    WT.iloc[i0 + 24 * d + self.hours[h]:i0 + 24 * d + self.hours[h] + tw, :], data1=data1,data2=data2,model=model,
                    predict=predict, **kwargs)
            
                #print(m)
                #print(np.size(data.get_stations_capacities(None)))
                m1 = m1 / data1.get_stations_capacities(None) #gerar a porcentagem
                t1 = t1 / data1.get_stations_capacities(None) #Aqui
                M1 = M1 / data1.get_stations_capacities(None)
                
                m2 = m2 / data2.get_stations_capacities(None) #gerar a porcentagem
                t2 = t2 / data2.get_stations_capacities(None) #Aqui
                M2 = M2 / data2.get_stations_capacities(None)
                
                #print ("Day "+str(d+1) + " and  perid "+str(h))
                #print(m)
                #self.hours = [6, 9, 11, 16, 19, 22]
                inter_min_re.to_numpy()[:, self.periods * d + h] = m1 # Mudar 
                inter_target_re.to_numpy()[:, self.periods * d + h] = t1
                inter_max_re.to_numpy()[:, self.periods * d + h] = M1


                inter_min_el.to_numpy()[:, self.periods * d + h] = m2 # Mudar 
                inter_target_el.to_numpy()[:, self.periods * d + h] = t2
                inter_max_el.to_numpy()[:, self.periods * d + h] = M2
                #print(inter_min.iloc[:, 6 * d + h])

        min_max1 =  pd.DataFrame(data=np.zeros((data1.get_stations_capacities(None).shape[0], (7 * self.periods*3)+1)), columns=['info.terminalName']+cols)
        min_max1['info.terminalName'] = data1.get_stations_ids(None)
        index_min_max = min_max1['info.terminalName']

        min_max2 =  pd.DataFrame(data=np.zeros((data2.get_stations_capacities(None).shape[0], (7 * self.periods*3)+1)), columns=['info.terminalName']+cols)
        min_max2['info.terminalName'] = data2.get_stations_ids(None)
        # index_min_max = min_max['info.terminalName']
        # print(min_max)
        
        def f(x):
            try:
                int(x)
                return int(x)
            except ValueError:
                return -1



        index_min_max = list(map(f, index_min_max))
        
        min_max1 = min_max1.set_index([index_min_max])
        min_max2 = min_max2.set_index([index_min_max])
        
        min_max1.update(inter_min_re * 100)
        min_max1.update(inter_max_re * 100)
        min_max1.update(inter_target_re * 100)
        
        min_max2.update(inter_min_el * 100)
        min_max2.update(inter_max_el * 100)
        min_max2.update(inter_target_el * 100)
        
        # print(min_max)
        #print(data.env.decision_intervals[:-4] + kwargs['distrib'] + '.csv')
        if save:
            # print("Inventory")
            # print(min_max)
            min_max1.to_csv(data1.env.decision_intervals[:-4] + '_avg_reg_'+str(self.param_beta1)+'_'+str(self.param_beta2)+'_'+str(ext)+'.csv', index=False)

            min_max2.to_csv(data2.env.decision_intervals[:-4] + '_avg_ele_'+str(self.param_beta1)+'_'+str(self.param_beta2)+'_'+str(ext)+'.csv', index=False)

        return min_max1,min_max2
    
    def compute_min_max_data2(self, WT, data1,data2, predict=True, save=True, **kwargs):
        """
        compute the min/max/target intervals starting from the first monday in WT and saves it to file 
        :param WT: features to predict on 
        :param data: the enviroment data (Data object) for station information
        :param predict: predict: if True use predictions, if False use real data
        :return: intervals data frame
        """
        # print(WT)
        # print(WT['Jour'])
        # hparam.update(kwargs)
        tw =5
        cond = True
        i0 = 0
        while cond:
            cond = not ((WT['wday'].to_numpy()[i0] == 6) and (WT['Heure'].to_numpy()[i0] == 0)) #meia noite do domingo
            i0 += 1
        i0 -= 1 # quanto horas ate segunda de manha

        cols_min = []
        cols_max = []
        cols_target = []
        cols = []
        for i in range(1, 8):
            for t in range(1, self.periods+1): #Mudar
                cols_min.append('dayOfWeek' + str(i) + '.period' + str(t) + 'Min')
                cols_target.append('dayOfWeek' + str(i) + '.period' + str(t) + 'Target')
                cols_max.append('dayOfWeek' + str(i) + '.period' + str(t) + 'Max')
                cols.append('dayOfWeek' + str(i) + '.period' + str(t) + 'Min')
                cols.append('dayOfWeek' + str(i) + '.period' + str(t) + 'Target')
                cols.append('dayOfWeek' + str(i) + '.period' + str(t) + 'Max')
        

        inter_min_re = pd.DataFrame(np.zeros((data1.get_stations_capacities(None).shape[0], 7 * self.periods)), #Mudar
                                 index=data1.get_stations_ids(None), columns=cols_min)
        inter_max_re = pd.DataFrame(np.zeros((data1.get_stations_capacities(None).shape[0], 7 * self.periods)), # Criando um dataFrame linhas sao as estações e colunas cols_min
                                 index=data1.get_stations_ids(None), columns=cols_max)
        inter_target_re = pd.DataFrame(np.zeros((data1.get_stations_capacities(None).shape[0], 7 * self.periods)),
                                    index=data1.get_stations_ids(None), columns=cols_target)
        inter_min_el = pd.DataFrame(np.zeros((data2.get_stations_capacities(None).shape[0], 7 * self.periods)), #Mudar
                                 index=data2.get_stations_ids(None), columns=cols_min)
        inter_max_el = pd.DataFrame(np.zeros((data2.get_stations_capacities(None).shape[0], 7 * self.periods)), # Criando um dataFrame linhas sao as estações e colunas cols_min
                                 index=data2.get_stations_ids(None), columns=cols_max)
        inter_target_el = pd.DataFrame(np.zeros((data2.get_stations_capacities(None).shape[0], 7 * self.periods)),
                                    index=data2.get_stations_ids(None), columns=cols_target)
        #Aqui
        
        for d in range(7): #dia da semana 
            for h in range(len(self.hours)): #perido
                # print("day "+str(d))
                # print("period "+str(h))
                # print(i0 + 24 * d + self.hours[h] + tw, end='\r')
                # print(WT.iloc[i0:])
                assert (WT['wday'].iloc[i0 + 24 * d + self.hours[h]] == (d - 1) % 7)

                m1, t1, M1, m2, t2, M2 = self.compute_decision_intervals2(WT=
                    WT.iloc[i0 + 24 * d + self.hours[h]:i0 + 24 * d + self.hours[h] + tw, :], data1=data1,data2=data2,
                    predict=predict, **kwargs)

                #print(m)
                #print(np.size(data.get_stations_capacities(None)))
                m1 = m1 / data1.get_stations_capacities(None) #gerar a porcentagem
                t1 = t1 / data1.get_stations_capacities(None) #Aqui
                M1 = M1 / data1.get_stations_capacities(None)
                
                m2 = m2 / data2.get_stations_capacities(None) #gerar a porcentagem
                t2 = t2 / data2.get_stations_capacities(None) #Aqui
                M2 = M2 / data2.get_stations_capacities(None)
                
                #print ("Day "+str(d+1) + " and  perid "+str(h))
                #print(m)
                #self.hours = [6, 9, 11, 16, 19, 22]
                inter_min_re.to_numpy()[:, self.periods * d + h] = m1 # Mudar 
                inter_target_re.to_numpy()[:, self.periods * d + h] = t1
                inter_max_re.to_numpy()[:, self.periods * d + h] = M1


                inter_min_el.to_numpy()[:, self.periods * d + h] = m2 # Mudar 
                inter_target_el.to_numpy()[:, self.periods * d + h] = t2
                inter_max_el.to_numpy()[:, self.periods * d + h] = M2
                #print(inter_min.iloc[:, 6 * d + h])

        min_max1 =  pd.DataFrame(data=np.zeros((data1.get_stations_capacities(None).shape[0], (7 * self.periods*3)+1)), columns=['info.terminalName']+cols)
        min_max1['info.terminalName'] = data1.get_stations_ids(None)
        index_min_max = min_max1['info.terminalName']

        min_max2 =  pd.DataFrame(data=np.zeros((data2.get_stations_capacities(None).shape[0], (7 * self.periods*3)+1)), columns=['info.terminalName']+cols)
        min_max2['info.terminalName'] = data2.get_stations_ids(None)
        # index_min_max = min_max['info.terminalName']
        # print(min_max)
        
        def f(x):
            try:
                int(x)
                return int(x)
            except ValueError:
                return -1



        index_min_max = list(map(f, index_min_max))
        
        min_max1 = min_max1.set_index([index_min_max])
        min_max2 = min_max2.set_index([index_min_max])
        
        min_max1.update(inter_min_re * 100)
        min_max1.update(inter_max_re * 100)
        min_max1.update(inter_target_re * 100)
        
        min_max2.update(inter_min_el * 100)
        min_max2.update(inter_max_el * 100)
        min_max2.update(inter_target_el * 100)
        
        # print(min_max)
        #print(data.env.decision_intervals[:-4] + kwargs['distrib'] + '.csv')
        if save:
            # print("Inventory")
            # print(min_max)
            min_max1.to_csv(data1.env.decision_intervals[:-4] + 'HRP.csv', index=False)

            min_max2.to_csv(data2.env.decision_intervals[:-4] + 'HEP.csv', index=False)

        return min_max1,min_max2

    def compute_min_max_data_period(self, WT, data, predict=True, save=True, day = -1, **kwargs):
        """
        compute the min/max/target intervals starting from the first monday in WT and saves it to file 
        :param WT: features to predict on 
        :param data: the enviroment data (Data object) for station information
        :param predict: predict: if True use predictions, if False use real data
        :return: intervals data frame"""

        tw = 1
        cond = True
        i0 = 0
        # now  =  datetime.datetime.now()

        # def  di_period(h):  #synthetic data
        #     if h < 8:
        #         return 7
        #     elif h < 9:
        #         return 8
        #     elif h < 10:
        #         return 9
        #     elif h < 11:
        #         return 10
        #     elif h < 12:
        #         return 11
        #     elif h < 13:
        #         return 12
        #     elif h < 14:
        #         return 13
        #     elif h < 15:
        #         return 14
        #     elif h < 16:
        #         return 15
        #     elif h < 17:
        #         return 16
        #     elif h < 18:
        #         return 17
        #     elif h < 19:
        #         return 18
        #     else:
        #         return 19

        # period = di_period(now.hour)

        # while cond:
        #     cond = not ((WT['wday'].to_numpy()[i0] == 6) and (WT['Heure'].to_numpy()[i0] == 0)) #meia noite do domingo
        #     i0 += 1
        # i0 -= 1 # quanto horas ate segunda de manha
        WT = WT.reset_index(drop=True)
        cols_min = []
        cols_max = []
        cols_target = []
        cols=[]
        #real data
        # wday = int(WT.loc[0,'wday'])
        # day = int(WT.loc[0,'Jour'])
        # month = int(WT.loc[0,'Mois'])
        # year = int(WT.loc[0,'Annee'])
        
        # print("wday")
        # print(wday)
        # for i in range(1, 8):
        for t in range(1, self.periods+1): #Mudar
            cols_min.append('period' + str(t) + 'Min')
            cols_target.append('period' + str(t) + 'Target')
            cols_max.append('period' + str(t) + 'Max')
            cols.append('period' + str(t) + 'Min')
            cols.append('period' + str(t) + 'Target')
            cols.append('period' + str(t) + 'Max')
            # cols_min.append('dayOfWeek' + str(i) + '.period' + str(5) + 'min')
            # cols_target.append('dayOfWeek' + str(i) + '.period' + str(5) + 'target')
            # cols_max.append('dayOfWeek' + str(i) + '.period' + str(5) + 'max')
        # cols_min
        # cols_max = []
        # cols_target = []
        # min_max = pd.DataFrame(np.zeros((data.get_stations_capacities(None).shape[0], (7 * self.periods *3)+1)), columns=['info.terminalName']+cols, index = data.get_stations_ids(None))
        # min_max['info.terminalName'] = data.get_stations_ids(None)
        # print(min_max)
        
        inter_min = pd.DataFrame(np.zeros((data.get_stations_capacities(None).shape[0], self.periods)), #Mudar
                                 index=data.get_stations_ids(None), columns=cols_min)
        inter_max = pd.DataFrame(np.zeros((data.get_stations_capacities(None).shape[0], self.periods)), # Criando um dataFrame linhas sao as estações e colunas cols_min
                                 index=data.get_stations_ids(None), columns=cols_max)
        inter_target = pd.DataFrame(np.zeros((data.get_stations_capacities(None).shape[0], self.periods)),
                                    index=data.get_stations_ids(None), columns=cols_target)
        
        #Aqui
        
        # for d in range(7): #dia da semana 
        for h in range(len(self.hours)): #perido
            
            # print(i0 + 24 * d + self.hours[h] + tw, end='\r')
            # assert (WT['wday'].iloc[i0 + 24 * d + self.hours[h]] == (d - 1) % 7)

            # if now.weekday()== d and period ==  h+1 :
            # if WT.iloc[self.hours[h]: self.hours[h] + tw, :].empty:
            #     break
            # print(WT)
            # print(WT.iloc[ self.hours[h]: self.hours[h] + tw, :])
            # return
            m, t, M = self.compute_decision_intervals(WT.iloc[ self.hours[h]: self.hours[h] + tw, :], data, predict, **kwargs)
            # m, t, M = self.compute_decision_intervals(WT.iloc[24 * d + self.hours[h]: 24 * d + self.hours[h] + tw, :], data, predict, **kwargs)
            # else:
            #     m = np.zeros(60)
            #     t = np.zeros(60)
            #     M = np.zeros(60)
            # print(m)

            # print(m)
            # print(type(m))
            # print(np.size(m))
            # print(np.size(data.get_stations_capacities(None)))
            m = m / data.get_stations_capacities(None) #gerar a porcentagem
            t = t / data.get_stations_capacities(None) #Aqui
            M = M / data.get_stations_capacities(None)
            
            # print(m)
            # print(type(m))
            # print(np.size(m))
            
            # print("capacity")
            # print(data.get_stations_capacities(None))
            #print ("Day "+str(d+1) + " and  perid "+str(h))
            #print(m)
            #self.hours = [6, 9, 11, 16, 19, 22]
            # min_max['dayOfWeek' + str(d+1) + '.period' + str(h+1) + 'Min'] = m *100
            # min_max['dayOfWeek' + str(d+1) + '.period' + str(h+1) + 'Min'] = t *100
            # min_max['dayOfWeek' + str(d+1) + '.period' + str(h+1) + 'Min'] = M *100
            inter_min.to_numpy()[:,  h] = m # Mudar 
            inter_target.to_numpy()[:,  h] = t
            inter_max.to_numpy()[:,  h] = M
            # print(inter_min)
                
        # print(min_max)

        #print(inter_min)
        # if  not self.real_data:
        #     min_max = pd.read_csv(open(config.root_path + 'bixi_intervals/intervals_syn_data.csv'), sep=',')
        # elif self.year == 2019:
        #     min_max = pd.read_csv(open(config.root_path + 'bixi_intervals/intervals_aug_jul_2019.csv'), sep=',') #Mudar
        # else:
        #     min_max = pd.read_csv(open(config.root_path + 'bixi_intervals/intervals_jun_2020.csv'), sep=',')
        # min_max[cols_min] = np.nan
        # min_max[cols_max] = np.nan
        # min_max[cols_target] = np.nan
        min_max =  pd.DataFrame(data=np.zeros((data.get_stations_capacities(None).shape[0], (self.periods*3)+1)), columns=['info.terminalName']+cols)
        min_max['info.terminalName'] = data.get_stations_ids(None)
        index_min_max = min_max['info.terminalName']
        # print(min_max)

        def f(x):
            try:
                int(x)
                return int(x)
            except ValueError:
                return -1



        index_min_max = list(map(f, index_min_max))
        min_max = min_max.set_index([index_min_max])
        min_max.update(inter_min * 100)
        min_max.update(inter_max * 100)
        min_max.update(inter_target * 100)
        
        # print("DIIII")
        # print(min_max)
        #print(data.env.decision_intervals[:-4] + kwargs['distrib'] + '.csv')
        if save:
            #print("OKKKK")
            if day>-1:
                min_max.to_csv('Bixi_decision_intervals/synthetic/DI_synthetic_data_day_'+str(day)+'_1.csv', index=False)
                # min_max.to_csv('Bixi_decision_intervals/real/DI_bixi_data_day_'+str(month)+'_'+str(day)+'_0.4.csv', index=False)
            else:
                min_max.to_csv('Bixi_decision_intervals/DI_'+str(now.day)+'_'+str(now.month)+'_period_'+str(period)+'.csv', index=False)
        #else:
            #print("NOOT")
        return min_max

    def general_min_max(self, WT, data, predict=True, **kwargs):
        """
        min max function independent of the weather

        To be tested and completed and improved (very slow)

        :param WT: features
        :param data: environment data (Data object)
        :param predict: if True use predictions, if False use real data
        :param kwargs:
        :return: interval dataframe
        """
        hparam = {
            'distrib': 'NB'
        }
        hparam.update(kwargs)
        min_max = None
        k = 0
        n = int(WT.shape[0] / 7 / 24)
        n = 2
        print()
        for k in range(0, n - 1):
            i = k * 24 * 7
            print(i)
            m = self.compute_min_max_data(WT[i:], data, predict, save=False, **hparam)
            if min_max is None:
                min_max = m
            else:
                min_max += m
        #min_max /= n
        min_max.to_csv(data.env.decision_intervals[:-4] + hparam['distrib'] + '.csv', index=False)
        print('end general int ' + hparam['distrib'])
        return min_max

    def transform_to_intervals(self, data, df):
        """
        read bixi intervals and reshape it
        :param data: environment
        :param df: bixi intervals dataframe
        :return: 3 dataframe containing the minimum, the maximum and the target for each station for each period
        """
        #print(df.head(2))
        df['code'] = df['info.terminalName']

        def f(x):
            try:
                return int(x)
            except:
                return -1

        df.loc[:, 'code'] = df['code'].apply(f)
        df.set_index(df['code'], inplace=True)
        #df.drop(-1, axis=0, inplace=True)
        # print(df)
        # print(list(df))
        df['info.capacity']=0
        df['info.capacity'].update(data.get_stations_capacities(None, df['code'].to_numpy()))
        
        rowsmin = {}
        rowsmax = {}
        rowstar = {}
        # renommer les lignes
        for i in range(1, 8):
            i_prime = ((i - 5) % 7)
            for t in range(1, self.periods+1): #Mudar
                rowsmin['dayOfWeek' + str(i) + '.period' + str(t) + 'Min'] = 24 * i_prime + self.hours[t - 1]  # +' Min'
                rowstar['dayOfWeek' + str(i) + '.period' + str(t) + 'Target'] = 24 * i_prime + self.hours[t - 1]
                rowsmax['dayOfWeek' + str(i) + '.period' + str(t) + 'Max'] = 24 * i_prime + self.hours[t - 1]  # +' Max'
            # rowsmin['dayOfWeek' + str(i) + '.period' + str(5) + 'min'] = 24 * i_prime + self.hours[4]  # +' Min'
            # rowstar['dayOfWeek' + str(i) + '.period' + str(5) + 'target'] = 24 * i_prime + self.hours[4]
            # rowsmax['dayOfWeek' + str(i) + '.period' + str(5) + 'max'] = 24 * i_prime + self.hours[4]  # +' Max'
        for val in [list(rowsmin.keys()), list(rowsmax.keys()), list(rowstar.keys())]:
            val.sort()
            df.loc[:, val] = mini(maxi(df.loc[:, val] / 100,0),1)
            for i in range(len(val)):
                df.loc[:, val[i]] = df.loc[:, val[i]] * df.loc[:, 'info.capacity']
            for _ in range(2):
                for i in range(len(val)):
                    df.loc[df[val[i]].isnull(), val[i]] = df[val[i - len(self.hours)]][df[val[i]].isnull()]
        #aqui
        dfmin = df[list(rowsmin.keys())]
        dfmax = df[list(rowsmax.keys())]
        dftar = df[list(rowstar.keys())]
        dfmin = dfmin.rename(columns=rowsmin)
        dfmax = dfmax.rename(columns=rowsmax)
        dftar = dftar.rename(columns=rowstar)
        dfmin = dfmin.transpose()
        dftar = dftar.transpose()
        dfmax = dfmax.transpose()
        #aqui
        rm = {}
        rM = {}
        rt = {}
        for col in dfmin.columns.values:
            rm[col] = 'Min ' + str(col)
            rM[col] = 'Max ' + str(col)
            rt[col] = 'Tar ' + str(col)
        dfmin.rename(columns=rm, inplace=True)
        dfmax.rename(columns=rM, inplace=True)
        dftar.rename(columns=rt, inplace=True)
        
        return dfmin, dfmax, dftar

    def load_intervals(self, data, path, distrib):
        """
        load decision intervals in a df hours, 3*nstations, 1 col max, min et target per station
        :param data: environment
        :return: df
        """
        path = path[:-4] + distrib + path[-4:]
        df = pd.read_csv(path, sep=',', encoding='latin-1')
        return self.transform_to_intervals(data, df)

    def load_intervals_wo_transform(self, data, path, distrib):
        """
        load decision intervals in a df hours, 3*nstations, 1 col max, min et target per station
        :param data: environment
        :return: df
        """
        path = path[:-4] + distrib + path[-4:]
        df = pd.read_csv(path, sep=',', encoding='latin-1')
        return df

    def compute_mean_intervals(self, data, path, distrib,stations=None):
        """
        computes the average length of intervals
        :param data: environment
        :param path: interval file
        :param distrib: distribution hypothesis
        :return: average length of intervals
        """
        path = path[:-4] + distrib + path[-4:]
        df = pd.read_csv(path, sep=',', encoding='latin-1')
        df['code'] = df['info.terminalName']

        def f(x):
            try:
                return int(x)
            except:
                return -1

        df.loc[:, 'code'] = df['code'].apply(f)
        df.set_index(df['code'], inplace=True)
        #df.drop(-1, axis=0, inplace=True)
        df['info.capacity'].update(data.get_stations_capacities(None, df['code'].to_numpy()))
        rowsmin = []
        rowsmax = []
        for i in range(1, 8):
            for t in range(1, self.periods+1): #Mudar
                rowsmin.append('dayOfWeek' + str(i) + '.period' + str(t) + 'Min')
                rowsmax.append('dayOfWeek' + str(i) + '.period' + str(t) + 'Max')
            # rowsmin.append('dayOfWeek' + str(i) + '.period' + str(5) + 'min')
            # rowsmax.append('dayOfWeek' + str(i) + '.period' + str(5) + 'max')
        for val in [list(rowsmin), list(rowsmax)]:
            val.sort()
            df.loc[:, val] = df.loc[:, val] / 100
            for i in range(len(val)):
                df.loc[:, val[i]] = df.loc[:, val[i]] * df.loc[:, 'info.capacity']
            for _ in range(2):
                for i in range(len(val)):
                    df.loc[df[val[i]].isnull(), val[i]] = df[val[i - len(self.hours)]][df[val[i]].isnull()]

        m = df[rowsmin].to_numpy()
        M = df[rowsmax].to_numpy()
        return np.nanmean(M - m)
   

    def alert_lost_demand_both_inv(self,test_data_re,test_data_el,intervals_re, intervals_el, df_re, df_el,reb_type,subs):
        # print('reb_type')
        # print(reb_type)
        interminr, intermaxr, intertarr = intervals_re
        intermine, intermaxe, intertare = intervals_el 
        
        cols_min = test_data_re.get_col('Min ',None)
        cols_max = test_data_re.get_col('Max ',None)
        cols_tar = test_data_re.get_col('Tar ',None)
        colstart = test_data_re.get_dep_cols(None) #star date
        colend   = test_data_re.get_arr_cols(None)

        cols_inve_re = test_data_re.get_col('Inve_reg_',None)
        cols_inve_el = test_data_el.get_col('Inve_ele_',None)
        cols_aler_re = test_data_re.get_col('Alert_reg_',None)
        cols_aler_el = test_data_el.get_col('Alert_ele_',None)
        # cols_reba = test_data.get_col('Reba_',None)
        cols_lost_re = test_data_re.get_col('Lost_reg_',None)
        cols_lost_el = test_data_el.get_col('Lost_ele_',None) 

        interminr = interminr[cols_min].astype(np.double) # intervalo de todas as estações (dataframe intervalos x estações)
        intermaxr = intermaxr[cols_max].astype(np.double)
        intertarr = intertarr[cols_tar].astype(np.double)
        intermine = intermine[cols_min].astype(np.double) 
        intermaxe = intermaxe[cols_max].astype(np.double)
        intertare = intertare[cols_tar].astype(np.double)
        
        interminr.rename(lambda x:int(x[4:]),axis=1,inplace=True) # renomear para numero da estação
        intermaxr.rename(lambda x:int(x[4:]),axis=1,inplace=True)
        intertarr.rename(lambda x:int(x[4:]),axis=1,inplace=True)
        intermine.rename(lambda x:int(x[4:]),axis=1,inplace=True) # renomear para numero da estação
        intermaxe.rename(lambda x:int(x[4:]),axis=1,inplace=True)
        intertare.rename(lambda x:int(x[4:]),axis=1,inplace=True)
        interminr = np.round(interminr)
        intertarr = np.round(intertarr)
        intermaxr = np.round(intermaxr)
        intermine = np.round(intermine)
        intertare = np.round(intertare)
        intermaxe = np.round(intermaxe) #(42 x 609) intervalos x estações
        df_re = df_re.reset_index()
        df_el = df_el.reset_index()
        
        mat_re = df_re[colend].to_numpy() - df_re[colstart].to_numpy()  # returns - rentals  (numpy horas x stations)
        mat_el = df_el[colend].to_numpy() - df_el[colstart].to_numpy()
        
        rent_re = df_re[colstart].to_numpy()
        rent_el = df_el[colstart].to_numpy()
        # print("antes")
        # print(rent_re)

        return_re = df_re[colend].to_numpy()
        return_el = df_el[colend].to_numpy()
        
        # rental_re = df_re[colstart].to_numpy()
        # rental_el = df_el[colstart].to_numpy()
        
        def f(h):  #2019
            if h < 6 or h>22: 
                return 6
            elif h < 9:
                return 9
            elif h < 11:
                return 11
            elif h < 16:
                return 16
            elif h < 19:
                return 19
            else:
                return 22
        
        features = copy.deepcopy(df_re['wday'] * 24 + df_re['Heure'].apply(f)) #vetor onde cada hora tem um numero que representa um intervalo  
        inv_re = copy.deepcopy(intertarr.loc[features[0], :].to_numpy()) 
        inv_el = copy.deepcopy(intertare.loc[features[0], :].to_numpy())  #inventorio de tds as estações no primeiro periodo comeca em ótimo (numpy array)
        cap = copy.deepcopy(test_data_re.get_stations_capacities(None)) #capacidade de cada estação
        lost_arr_re = copy.deepcopy(np.zeros(mat_re.shape[1])) #shape stations
        lost_dep_re = copy.deepcopy(np.zeros(mat_re.shape[1]))
        lost_arr_el = copy.deepcopy(np.zeros(mat_el.shape[1]))
        lost_dep_el = copy.deepcopy(np.zeros(mat_el.shape[1]))

        alert_arr_re = copy.deepcopy(np.zeros(mat_re.shape[1]))
        alert_dep_re = copy.deepcopy(np.zeros(mat_re.shape[1]))
        alert_arr_el = copy.deepcopy(np.zeros(mat_el.shape[1]))
        alert_dep_el = copy.deepcopy(np.zeros(mat_el.shape[1]))
        
        stations_ = test_data_re.get_stations_ids()
        # print(stations_)
        # qt_stations_rb = 0
        empty_stations = 0
        full_stations = 0
        rebalanced = 0
        df_result = pd.DataFrame([])
        # print(reb_type)
        # return
        for i in range(mat_re.shape[0]-1): #per hour
           
            inv_re = copy.deepcopy(inv_re + mat_re[i, :]) # mudando o inventorio de acordo com a demanda
            inv_el = copy.deepcopy(inv_el + mat_el[i, :])
            

            if subs == 1:
                
                for e,j in enumerate(inv_el): 
                    if j<0 and rent_el[i, e]>0 and inv_re[e]>0:
                        for m in np.arange(-j):
                            # if random.randint(0, 1) == 1:
                            inv_re[e] = copy.deepcopy(inv_re[e] - 1)
                            inv_el[e] = copy.deepcopy(inv_el[e] + 1)
                            
                            if inv_re[e]<=0:
                                break
                
                for e,j in enumerate(inv_re): 
                    if j<0 and rent_re[i, e]>0 and inv_el[e]>0:
                        for m in np.arange(-j):
                            # if random.randint(0, 1) == 1:
                            inv_re[e] = copy.deepcopy(inv_re[e] + 1)
                            inv_el[e] = copy.deepcopy(inv_el[e] - 1)
                            
                            if inv_el[e]<=0:
                                break
                
                
            elif subs == 2:
                
                for e,j in enumerate(inv_re): 
                    if j<0 and rent_re[i, e]>0 and inv_el[e]>0:
                        for m in np.arange(-j):
                            # if random.randint(0, 1) == 1:
                                inv_re[e] = copy.deepcopy(inv_re[e] + 1)
                                inv_el[e] = copy.deepcopy(inv_el[e] - 1)

                                if inv_el[e]<=0:
                                    break
            elif subs==3:    
                for e,j in enumerate(inv_el): 
                    if j<0 and rent_el[i, e]>0 and inv_re[e]>0:
                        for m in np.arange(-j):
                            # if random.randint(0, 1) == 1:
                                inv_re[e] = copy.deepcopy(inv_re[e] - 1)
                                inv_el[e] = copy.deepcopy(inv_el[e] + 1)
                                
                                if inv_re[e]<=0:
                                    break
            

            inv_to = copy.deepcopy(np.where(inv_re<0, 0, inv_re)+np.where(inv_el<0, 0, inv_el))
            empty_stations = empty_stations + sum((inv_to==0))
            full_stations = full_stations + sum(((inv_to)==cap))

            # df_result.at[i,"date"] = df_re['Date/Heure'].iloc[i]
            # df_result.at[i,cols_inve_re] = inv_re
            # df_result.at[i,cols_inve_el] = inv_el
            

            alert_arr_re = alert_arr_re + (inv_re<interminr.loc[features[i],:].to_numpy()).tolist() #somando os alertas de todas as estações
            alert_dep_re = alert_dep_re + (inv_re>intermaxr.loc[features[i],:].to_numpy()).tolist()

            alert_arr_el = alert_arr_el + (inv_el<intermine.loc[features[i],:].to_numpy()).tolist() #somando os alertas de todas as estações
            alert_dep_el = alert_dep_el + (inv_el>intermaxe.loc[features[i],:].to_numpy()).tolist()
            
            # df_result.at[i,cols_aler_re] = (inv_re<interminr.loc[features[i],:].to_numpy()) | (inv_re>intermaxr.loc[features[i],:].to_numpy())
            # df_result.at[i,cols_aler_el] = (inv_el<intermine.loc[features[i],:].to_numpy()) | (inv_el>intermaxe.loc[features[i],:].to_numpy())
            
            # df_result.at[i,"alerts"] = sum((inv_re<interminr.loc[features[i],:].to_numpy()).tolist())+sum((inv_re>intermaxr.loc[features[i],:].to_numpy()).tolist())+sum((inv_el<intermine.loc[features[i],:].to_numpy()).tolist())+sum((inv_el>intermaxe.loc[features[i],:].to_numpy()).tolist())

            lost_arr_re = lost_arr_re - np.where(inv_re>0, 0, inv_re).tolist()
            lost_arr_el = lost_arr_el - np.where(inv_el>0, 0, inv_el).tolist()
            
            # df_result.at[i,cols_lost_re] = np.where(inv_re>0, 0, -inv_re).tolist()
            # df_result.at[i,cols_lost_el] = np.where(inv_el>0, 0, -inv_el).tolist()

            np.seterr(invalid='ignore')
           
            
            
            lost = (inv_to-cap)
            lost = np.where(lost<0, 0, lost)

            # pro_re = np.nan_to_num((abs(mat_re[i, :])*0.75)/(abs(mat_re[i, :])+abs(mat_el[i, :])))
            # pro_el = np.nan_to_num((abs(mat_el[i, :])*0.25)/(abs(mat_re[i, :])+abs(mat_el[i, :])))
           
            # lost_re = [int(Decimal(x).quantize(0, ROUND_HALF_DOWN)) for x in np.nan_to_num((lost * pro_re)/(pro_re+pro_el))]
            # lost_el = [int(Decimal(x).quantize(0, ROUND_HALF_UP))   for x in np.nan_to_num((lost * pro_el)/(pro_re+pro_el))]
           
            # print()
            lost_re = copy.deepcopy(np.round(np.nan_to_num((lost*return_re[i, :])/(return_re[i, :]+return_el[i, :]))))
            lost_el = copy.deepcopy(lost - lost_re)

            lost_dep_re = copy.deepcopy(lost_dep_re + lost_re)
            lost_dep_el = copy.deepcopy(lost_dep_el + lost_el)
            

            # if reb_type == '1':
            list_alerts_re = [a or b for a, b in zip((inv_re<interminr.loc[features[i],:].to_numpy()), (inv_re>intermaxr.loc[features[i],:].to_numpy()))]
            list_alerts_el = [a or b for a, b in zip((inv_el<intermine.loc[features[i],:].to_numpy()), (inv_el>intermaxe.loc[features[i],:].to_numpy()))]
            list_alerts = [ True if a or b else False for (a, b) in zip(list_alerts_re, list_alerts_el)]
            stations_alerts = [q for (q, v) in zip(stations_, list_alerts) if v]
            
            # print(type(stations_alerts))
            # print(type(stations_alerts))
            # print(len(stations_alerts))
            # # print(reb_type[i])
            # print(random.sample(stations_alerts, int(reb_type[i])).sort())
            # print(len(random.sample(stations_alerts, int(reb_type[i]))))
            
            #######################################################################
            if len(stations_alerts)>int(reb_type[i]):
                stations_ind = [list(stations_).index(x) for x in stations_alerts]
                pri1 = abs(inv_re[stations_ind] - intertarr.loc[features[i],stations_alerts])
                pri2 = abs(inv_el[stations_ind] - intertare.loc[features[i],stations_alerts])
                pri = pri1+pri2
                pri.sort_values(ascending=False,inplace=True)
                stations_alerts =  pri.index[:int(reb_type[i])].tolist()
                stations_alerts.sort()
            #     # print(stations_alerts)
            #     # return
                # stations_alerts = random.sample(stations_alerts, int(reb_type[i]))
                # stations_alerts.sort()
            #########################################################################
            # print(stations_alerts)
            # print(len(stations_alerts))
            # return

            df_result.at[i,'rebalanced'] = len(stations_alerts)
            
            stations_ind = [list(stations_).index(x) for x in stations_alerts]
            
            rebalanced = rebalanced + len(stations_alerts)

            inv_re[stations_ind] = intertarr.loc[features[i],stations_alerts] 
            inv_re = np.round(inv_re,0)
            
            inv_el[stations_ind] = intertare.loc[features[i],stations_alerts] 
            inv_el = np.round(inv_el,0)

            inv_re = copy.deepcopy(np.where(inv_re<0, 0, inv_re)) # removendo inventario fantasma
            inv_el = copy.deepcopy(np.where(inv_el<0, 0, inv_el))
            
            # print(lost_re)
            inv_re = copy.deepcopy(inv_re-lost_re) # removendo inventario fantasma
            inv_el = copy.deepcopy(inv_el-lost_el) 

            # else:
            #     list_alerts_re = [a or b for a, b in zip((inv_re<=interminr.loc[features[i],:].to_numpy()), (inv_re>=intermaxr.loc[features[i],:].to_numpy()))]
            #     list_alerts_el = [a or b for a, b in zip((inv_el<=intermine.loc[features[i],:].to_numpy()), (inv_el>=intermaxe.loc[features[i],:].to_numpy()))]
            #     list_alerts = [ True if a or b else False for (a, b) in zip(list_alerts_re, list_alerts_el)]
            #     stations_alerts_re = [q for (q, v) in zip(stations_, list_alerts_re) if v]
            #     stations_alerts_el = [q for (q, v) in zip(stations_, list_alerts_el) if v]
            #     rebalanced = rebalanced + sum(list_alerts)

            #     stations_ind_re = [list(stations_).index(x) for x in stations_alerts_re]
            #     inv_re[stations_ind_re] = intertarr.loc[features[i],stations_alerts_re] 
            #     inv_re = np.round(inv_re,0)
                
            #     stations_ind_el = [list(stations_).index(x) for x in stations_alerts_el]
            #     inv_el[stations_ind_el] = intertare.loc[features[i],stations_alerts_el] 
            #     inv_el = np.round(inv_el,0)
        
        return df_result
        return sum(lost_arr_re)/sum(sum(rent_re)), sum(lost_dep_re)/sum(sum(return_re)),sum(lost_arr_el)/sum(sum(rent_el)), sum(lost_dep_el)/sum(sum(return_el)),lost_arr_re.sum(),lost_dep_re.sum(),lost_arr_el.sum(),lost_dep_el.sum(),rebalanced,empty_stations,full_stations   
        # return alert_arr_re.sum(), alert_dep_re.sum(),alert_arr_el.sum(), alert_dep_el.sum(),lost_arr_re.sum(),lost_dep_re.sum(),lost_arr_el.sum(),lost_dep_el.sum(),rebalanced,empty_stations,full_stations
    
    def alert_lost_demand_bixi_inv(self,test_data_re,test_data_el,intervals, df_re, df_el,subs):
        intermin, intermax, intertar = intervals 
        
        stations_propor_reg = test_data_re.get_stations_proportions_reg()
        stations_propor_ele = np.full((len(stations_propor_reg)), 1) - stations_propor_reg
        # print(stations_propor_reg)
        # print(stations_propor_ele)
        # return
        cols_min = test_data_re.get_col('Min ',None)
        cols_max = test_data_re.get_col('Max ',None)
        cols_tar = test_data_re.get_col('Tar ',None)
        colstart = test_data_re.get_dep_cols(None) #star date
        colend   = test_data_re.get_arr_cols(None)
        cols_inve_re = test_data_re.get_col('Inve_reg_',None)
        cols_inve_el = test_data_el.get_col('Inve_ele_',None)
        cols_aler = test_data_re.get_col('Alert_',None)
        # cols_aler_el = test_data_el.get_col('Alert_ele_',None)
        # cols_reba = test_data.get_col('Reba_',None)
        cols_lost_re = test_data_re.get_col('Lost_reg_',None)
        cols_lost_el = test_data_el.get_col('Lost_ele_',None)  

        intermin = intermin[cols_min].astype(np.double) # intervalo de todas as estações (dataframe intervalos x estações)
        intermax = intermax[cols_max].astype(np.double)
        intertar = intertar[cols_tar].astype(np.double)
        
        intermin.rename(lambda x:int(x[4:]),axis=1,inplace=True) # renomear para numero da estação
        intermax.rename(lambda x:int(x[4:]),axis=1,inplace=True)
        intertar.rename(lambda x:int(x[4:]),axis=1,inplace=True)
        
        intermin = np.round(intermin)
        intertar = np.round(intertar)
        intermax = np.round(intermax) #(42 x 609) intervalos x estações

        df_re = df_re.reset_index()
        df_el = df_el.reset_index()
        
        mat_re = df_re[colend].to_numpy() - df_re[colstart].to_numpy()  # returns - rentals  (numpy horas x stations)
        mat_el = df_el[colend].to_numpy() - df_el[colstart].to_numpy()
        
        rent_re = df_re[colstart].to_numpy()
        rent_el = df_el[colstart].to_numpy()
        # print(intertar)
        # print(stations_propor_reg)
        # print(np.any(intertar < 0))
        # print(np.any(stations_propor_reg < 0))
        # return
        # print(df_re[colstart])
        # print(rent_re)
        # print(sum(rent_re))
        # print(len(sum(rent_re)))
        # return
        return_re = df_re[colend].to_numpy()
        return_el = df_el[colend].to_numpy()
        
        
        def f(h):  #2019
            if h < 6 or h>22: 
                return 6
            elif h < 9:
                return 9
            elif h < 11:
                return 11
            elif h < 16:
                return 16
            elif h < 19:
                return 19
            else:
                return 22
        
        features = copy.deepcopy(df_re['wday'] * 24 + df_re['Heure'].apply(f)) #vetor onde cada hora tem um numero que representa um intervalo  
        inv_re = copy.deepcopy(np.round(intertar.loc[features[0], :].to_numpy()*stations_propor_reg)) 
        inv_el = copy.deepcopy(np.round(intertar.loc[features[0], :].to_numpy()*stations_propor_ele))  #inventorio de tds as estações no primeiro periodo comeca em ótimo (numpy array)
        cap = copy.deepcopy(test_data_re.get_stations_capacities(None)) #capacidade de cada estação
        # print(intertar.loc[features[0], :])
        # print(inv_re)
        # print(inv_el)
        # return
        lost_arr_re = copy.deepcopy(np.zeros(mat_re.shape[1])) #shape stations
        lost_dep_re = copy.deepcopy(np.zeros(mat_re.shape[1]))
        lost_arr_el = copy.deepcopy(np.zeros(mat_el.shape[1]))
        lost_dep_el = copy.deepcopy(np.zeros(mat_el.shape[1]))

        alert_arr = copy.deepcopy(np.zeros(mat_re.shape[1]))
        alert_dep = copy.deepcopy(np.zeros(mat_re.shape[1]))
        # alert_arr_el = copy.deepcopy(np.zeros(mat_el.shape[1]))
        # alert_dep_el = copy.deepcopy(np.zeros(mat_el.shape[1]))
        
        stations_ = test_data_re.get_stations_ids()
        # print(stations_)
        empty_stations = 0
        full_stations = 0
        rebalanced = 0
        df_result = pd.DataFrame([])



        for i in range(mat_re.shape[0]-1): #per hour
           
            inv_re = copy.deepcopy(inv_re + mat_re[i, :]) # mudando o inventorio de acordo com a demanda
            inv_el = copy.deepcopy(inv_el + mat_el[i, :])

            # print(i)           
            if subs == 1:
                
                for e,j in enumerate(inv_el): 
                    if j<0 and rent_el[i, e]>0 and inv_re[e]>0:
                        for m in np.arange(-j):
                            # if random.randint(0, 1) == 1:
                            inv_re[e] = copy.deepcopy(inv_re[e] - 1)
                            inv_el[e] = copy.deepcopy(inv_el[e] + 1)
                            
                            if inv_re[e]<=0:
                                break
                
                for e,j in enumerate(inv_re): 
                    if j<0 and rent_re[i, e]>0 and inv_el[e]>0:
                        for m in np.arange(-j):
                            # if random.randint(0, 1) == 1:
                            inv_re[e] = copy.deepcopy(inv_re[e] + 1)
                            inv_el[e] = copy.deepcopy(inv_el[e] - 1)
                            
                            if inv_el[e]<=0:
                                break
                
                
            elif subs == 2:
                
                for e,j in enumerate(inv_re): 
                    if j<0 and rent_re[i, e]>0 and inv_el[e]>0:
                        for m in np.arange(-j):
                            # if random.randint(0, 1) == 1:
                                inv_re[e] = copy.deepcopy(inv_re[e] + 1)
                                inv_el[e] = copy.deepcopy(inv_el[e] - 1)

                                if inv_el[e]<=0:
                                    break
            elif subs==3:    
                for e,j in enumerate(inv_el): 
                    if j<0 and rent_el[i, e]>0 and inv_re[e]>0:
                        for m in np.arange(-j):
                            # if random.randint(0, 1) == 1:
                                inv_re[e] = copy.deepcopy(inv_re[e] - 1)
                                inv_el[e] = copy.deepcopy(inv_el[e] + 1)
                                
                                if inv_re[e]<=0:
                                    break
            # df_result.at[i,"date"] = df_re['Date/Heure'].iloc[i]
            # df_result.at[i,cols_inve_re] = inv_re
            # df_result.at[i,cols_inve_el] = inv_el
            

            inv_to = copy.deepcopy(np.where(inv_re<0, 0, inv_re)+np.where(inv_el<0, 0, inv_el))
            inv_to2 = copy.deepcopy(inv_re+inv_el)
            

            empty_stations = empty_stations + np.count_nonzero(inv_to==0)
            full_stations = full_stations + np.sum(inv_to==cap)
            
            alert_arr = alert_arr + ((inv_to<intermin.loc[features[i],:].to_numpy())).tolist() #somando os alertas de todas as estações
            alert_dep = alert_dep + ((inv_to>intermax.loc[features[i],:].to_numpy())).tolist()
            # df_result.at[i,'alerts_arr'] = sum(((inv_to<intermin.loc[features[i],:].to_numpy()) | (inv_to==0)).tolist())
            # df_result.at[i,'alerts_dep'] = sum(((inv_to>intermax.loc[features[i],:].to_numpy())).tolist())
            # df_result.at[i,cols_aler_re] = (inv_to<intermin.loc[features[i],:].to_numpy()) |  (inv_to>intermax.loc[features[i],:].to_numpy())

            lost_arr_re = lost_arr_re - np.where(inv_re>0, 0, inv_re).tolist()
            lost_arr_el = lost_arr_el - np.where(inv_el>0, 0, inv_el).tolist()
            # print(inv_el)
            # print(- np.where(inv_el>0, 0, inv_el).tolist())
            # print(np.where(inv_re>0, 0, inv_re).tolist())
            # print(lost_arr_re)
            # return             
            # df_result.at[i,cols_lost_re] = np.where(inv_re>0, 0, -inv_re).tolist()
            # df_result.at[i,"alerts"] = sum((inv_to<intermin.loc[features[i],:].to_numpy()).tolist()| (inv_to==0))+sum((inv_to>intermax.loc[features[i],:].to_numpy()).tolist())
            # df_result.at[i,"rentals_re"] = sum(rent_re[i, :]) # sum(mat_el[i, :])
            # df_result.at[i,"rentals_el"] = sum(rent_el[i, :]) # sum(mat_re[i, :])
            # df_result.at[i,"returns_re"] = sum(return_re[i, :]) 
            # df_result.at[i,"returns_el"] = sum(return_el[i, :]) 
            


            np.seterr(invalid='ignore')
           
            
            
            lost = (inv_to-cap)
            lost = np.where(lost<0, 0, lost)

            lost_re = np.round(np.nan_to_num((lost*return_re[i, :])/(return_re[i, :]+return_el[i, :])))
            lost_el = lost - lost_re

            lost_dep_re = lost_dep_re + lost_re
            lost_dep_el = lost_dep_el + lost_el
            

            list_alerts = [a or b for a, b in zip(((inv_to<intermin.loc[features[i],:].to_numpy())|(inv_to==0)), (inv_to>intermax.loc[features[i],:].to_numpy()))]
            # list_alerts_el = [a or b for a, b in zip((inv_el<intermine.loc[features[i],:].to_numpy()), (inv_el>intermaxe.loc[features[i],:].to_numpy()))]
            # list_alerts = [ True if a or b else False for (a, b) in zip(list_alerts_re, list_alerts_el)]
            stations_alerts = [q for (q, v) in zip(stations_, list_alerts) if v]
            stations_pro_reg = [q for (q, v) in zip(stations_propor_reg, list_alerts) if v]
            stations_pro_ele = [q for (q, v) in zip(stations_propor_ele, list_alerts) if v]
            
            rebalanced = rebalanced + sum(list_alerts)
            
            df_result.at[i,'rebalanced'] = sum(list_alerts)
            # df_result[i,'inv_reg'] = sum(list_alerts)

            stations_ind = [list(stations_).index(x) for x in stations_alerts]
            
            inv_re[stations_ind] = copy.deepcopy(np.round(intertar.loc[features[i],stations_alerts]*stations_pro_reg)) 
            inv_re = copy.deepcopy(np.round(inv_re,0))
            inv_el[stations_ind] = copy.deepcopy(np.round(intertar.loc[features[i],stations_alerts]*stations_pro_ele)) 
            inv_el = copy.deepcopy(np.round(inv_el,0))
            

            inv_re = copy.deepcopy(np.where(inv_re<0, 0, inv_re)) # removendo inventario fantasma
            inv_el = copy.deepcopy(np.where(inv_el<0, 0, inv_el))
           
            # inv_re = copy.deepcopy(np.array([ for a,b, in zip(inv_re)])) # mudando o inventorio de acordo com a demanda
            # inv_el = copy.deepcopy(np.where(inv_el:, 0, inv_el))
            # print(np.round(intertar.loc[features[i],stations_alerts]*stations_pro_ele))
        # print(rebalanced)
        # print(lost_arr_re.sum()+lost_dep_re.sum()+lost_arr_el.sum()+lost_dep_el.sum())
        # print(df_result)
        # return
        return df_result
        # return alert_arr.sum(), alert_dep.sum(),lost_arr_re.sum(),lost_dep_re.sum(),lost_arr_el.sum(),lost_dep_el.sum(),rebalanced,empty_stations,full_stations
        return sum(lost_arr_re)/sum(sum(rent_re)), sum(lost_dep_re)/sum(sum(return_re)),sum(lost_arr_el)/sum(sum(rent_el)), sum(lost_dep_el)/sum(sum(return_el)),lost_arr_re.sum(),lost_dep_re.sum(),lost_arr_el.sum(),lost_dep_el.sum(),rebalanced,empty_stations,full_stations 
        # print("aqui")
        # np.set_printoptions(suppress=True)
        # print(lost_arr_re)
        # print(sum(lost_arr_re))
        # print(lost_arr_re>sum(rent_re))
        # print(sum(lost_arr_re)/sum(sum(rent_re)))
        # return 
    def alert_lost_demand_hulot_inv(self,test_data_re,test_data_el,intervals, df_re, df_el,reb_type,subs):
        intermin, intermax, intertar = intervals 
        
        stations_propor_reg = test_data_re.get_stations_proportions_reg()
        stations_propor_ele = np.full((len(stations_propor_reg)), 1) - stations_propor_reg
        # print(stations_propor_reg)
        # print(stations_propor_ele)
        # return
        cols_min = test_data_re.get_col('Min ',None)
        cols_max = test_data_re.get_col('Max ',None)
        cols_tar = test_data_re.get_col('Tar ',None)
        colstart = test_data_re.get_dep_cols(None) #star date
        colend   = test_data_re.get_arr_cols(None)
        cols_inve_re = test_data_re.get_col('Inve_reg_',None)
        cols_inve_el = test_data_el.get_col('Inve_ele_',None)
        cols_aler = test_data_re.get_col('Alert_',None)
        # cols_aler_el = test_data_el.get_col('Alert_ele_',None)
        # cols_reba = test_data.get_col('Reba_',None)
        cols_lost_re = test_data_re.get_col('Lost_reg_',None)
        cols_lost_el = test_data_el.get_col('Lost_ele_',None)  

        intermin = intermin[cols_min].astype(np.double) # intervalo de todas as estações (dataframe intervalos x estações)
        intermax = intermax[cols_max].astype(np.double)
        intertar = intertar[cols_tar].astype(np.double)
        
        intermin.rename(lambda x:int(x[4:]),axis=1,inplace=True) # renomear para numero da estação
        intermax.rename(lambda x:int(x[4:]),axis=1,inplace=True)
        intertar.rename(lambda x:int(x[4:]),axis=1,inplace=True)
        
        intermin = np.round(intermin)
        intertar = np.round(intertar)
        intermax = np.round(intermax) #(42 x 609) intervalos x estações

        df_re = df_re.reset_index()
        df_el = df_el.reset_index()
        
        mat_re = df_re[colend].to_numpy() - df_re[colstart].to_numpy()  # returns - rentals  (numpy horas x stations)
        mat_el = df_el[colend].to_numpy() - df_el[colstart].to_numpy()
        
        rent_re = df_re[colstart].to_numpy()
        rent_el = df_el[colstart].to_numpy()

        return_re = df_re[colend].to_numpy()
        return_el = df_el[colend].to_numpy()
        
        
        def f(h):  #2019
            if h < 6 or h>22: 
                return 6
            elif h < 9:
                return 9
            elif h < 11:
                return 11
            elif h < 16:
                return 16
            elif h < 19:
                return 19
            else:
                return 22
        
        features = copy.deepcopy(df_re['wday'] * 24 + df_re['Heure'].apply(f)) #vetor onde cada hora tem um numero que representa um intervalo  
        inv_re = copy.deepcopy(np.round(intertar.loc[features[0], :].to_numpy()*stations_propor_reg)) 
        inv_el = copy.deepcopy(np.round(intertar.loc[features[0], :].to_numpy()*stations_propor_ele))  #inventorio de tds as estações no primeiro periodo comeca em ótimo (numpy array)
        cap = copy.deepcopy(test_data_re.get_stations_capacities(None)) #capacidade de cada estação
        # print(intertar.loc[features[0], :])
        # print(inv_re)
        # print(inv_el)
        # return
        lost_arr_re = copy.deepcopy(np.zeros(mat_re.shape[1])) #shape stations
        lost_dep_re = copy.deepcopy(np.zeros(mat_re.shape[1]))
        lost_arr_el = copy.deepcopy(np.zeros(mat_el.shape[1]))
        lost_dep_el = copy.deepcopy(np.zeros(mat_el.shape[1]))

        alert_arr = copy.deepcopy(np.zeros(mat_re.shape[1]))
        alert_dep = copy.deepcopy(np.zeros(mat_re.shape[1]))
        # alert_arr_el = copy.deepcopy(np.zeros(mat_el.shape[1]))
        # alert_dep_el = copy.deepcopy(np.zeros(mat_el.shape[1]))
        
        stations_ = test_data_re.get_stations_ids()
        
        rebalanced = 0
        # df_result = pd.DataFrame([])
        # print(reb_type)
        empty_stations = 0
        full_stations = 0

        for i in range(mat_re.shape[0]-1): #per hour
           
            inv_re += mat_re[i, :] # mudando o inventorio de acordo com a demanda
            inv_el += mat_el[i, :]

            if subs == 1:
                
                for e,j in enumerate(inv_re): 
                    if j<0 and rent_re[i, e]>0 and inv_el[e]>0:
                        for m in np.arange(-j):
                            if random.randint(0, 1) == 1:
                                inv_re[e] +=  1
                                inv_el[e] += -1

                                if inv_el[e]<=0:
                                    break
                
                for e,j in enumerate(inv_el): 
                    if j<0 and rent_el[i, e]>0 and inv_re[e]>0:
                        for m in np.arange(-j):
                            if random.randint(0, 1) == 1:
                                inv_el[e] +=  1
                                inv_re[e] += -1
                                if inv_re[e]<=0:
                                    break

            # df_result.at[i,"date"] = df_re['Date/Heure'].iloc[i]
            # df_result.at[i,cols_inve_re] = inv_re
            # df_result.at[i,cols_inve_el] = inv_el
            

            inv_to = np.where(inv_re<0, 0, inv_re)+np.where(inv_el<0, 0, inv_el)

            empty_stations = empty_stations + np.count_nonzero(inv_to==0)
            full_stations = full_stations + np.sum(inv_to==cap)
            
            alert_arr = alert_arr + ((inv_to<intermin.loc[features[i],:].to_numpy())|(inv_to==0)).tolist() #somando os alertas de todas as estações
            alert_dep = alert_dep + (inv_to>intermax.loc[features[i],:].to_numpy()).tolist()

            # df_result.at[i,cols_aler_re] = (inv_to<intermin.loc[features[i],:].to_numpy()) |  (inv_to>intermax.loc[features[i],:].to_numpy())

            lost_arr_re = lost_arr_re - np.where(inv_re>0, 0, inv_re).tolist()
            lost_arr_el = lost_arr_el - np.where(inv_el>0, 0, inv_el).tolist()
            # df_result.at[i,cols_lost_re] = np.where(inv_re>0, 0, -inv_re).tolist()
            # df_result.at[i,cols_lost_el] = np.where(inv_el>0, 0, -inv_el).tolist()

            np.seterr(invalid='ignore')
           
            
            
            lost = (inv_to-cap)
            lost = np.where(lost<0, 0, lost)

            # pro_re = np.nan_to_num((abs(mat_re[i, :])*0.75)/(abs(mat_re[i, :])+abs(mat_el[i, :])))
            # pro_el = np.nan_to_num((abs(mat_el[i, :])*0.25)/(abs(mat_re[i, :])+abs(mat_el[i, :])))
           
            # lost_re = [int(Decimal(x).quantize(0, ROUND_HALF_DOWN)) for x in np.nan_to_num((lost * pro_re)/(pro_re+pro_el))]
            # lost_el = [int(Decimal(x).quantize(0, ROUND_HALF_UP))   for x in np.nan_to_num((lost * pro_el)/(pro_re+pro_el))]
           
            # print()
            lost_re = np.round(np.nan_to_num((lost*return_re[i, :])/(return_re[i, :]+return_el[i, :])))
            lost_el = lost - lost_re

            lost_dep_re = lost_dep_re + lost_re
            lost_dep_el = lost_dep_el + lost_el
            


            list_alerts = [a or b for a, b in zip(((inv_to<intermin.loc[features[i],:].to_numpy())| (inv_to==0)), (inv_to>intermax.loc[features[i],:].to_numpy()))]
            # list_alerts_el = [a or b for a, b in zip((inv_el<intermine.loc[features[i],:].to_numpy()), (inv_el>intermaxe.loc[features[i],:].to_numpy()))]
            # list_alerts = [ True if a or b else False for (a, b) in zip(list_alerts_re, list_alerts_el)]
            stations_alerts = [q for (q, v) in zip(stations_, list_alerts) if v]
            
            # print(stations_alerts)
            # print(int(reb_type[i]))
            # print(stations_propor_reg)

            if len(stations_alerts)>int(reb_type[i]):
                #target
                # stations_ind = [list(stations_).index(x) for x in stations_alerts]
                # pri = abs(inv_to[stations_ind]-intertar.loc[features[i],stations_alerts])
                # pri.sort_values(ascending=False,inplace=True)
                # stations_alerts =  pri.index[:int(reb_type[i])].tolist()
                # stations_alerts.sort()

                #random
                stations_alerts = random.sample(stations_alerts, int(reb_type[i]))
                stations_alerts.sort()
                list_alerts = [True if v in stations_alerts else False for v in stations_]
                
            stations_pro_reg = [q for (q, v) in zip(stations_propor_reg, list_alerts) if v]
            stations_pro_ele = [q for (q, v) in zip(stations_propor_ele, list_alerts) if v]
            

            

            rebalanced = rebalanced + sum(list_alerts)
            # df_result.at[i,'rebalanced'] = sum(list_alerts)
            # df_result[i,'inv_reg'] = sum(list_alerts)
            stations_ind = [list(stations_).index(x) for x in stations_alerts]
            
            inv_re[stations_ind] = np.round(intertar.loc[features[i],stations_alerts]*stations_pro_reg) 
            inv_re = np.round(inv_re,0)
            
            inv_el[stations_ind] = np.round(intertar.loc[features[i],stations_alerts]*stations_pro_ele) 
            inv_el = np.round(inv_el,0)
        
        return alert_arr.sum(), alert_dep.sum(),lost_arr_re.sum(),lost_dep_re.sum(),lost_arr_el.sum(),lost_dep_el.sum(),rebalanced,empty_stations,full_stations    
        # return df_result
    
if __name__ == '__main__':
    ud_test = Environment('Bixi', 'test')
    test = Data(ud_test)
    ud_train = Environment('Bixi', 'train')
    train = Data(ud_train)
    ud_validation = Environment('Bixi', 'validation')
    validation = Data(ud_validation)
    
    # print(Data(ud_test).get_stations_col(None))
    mod = ModelStations(ud_train, 'svd', 'gbt', dim=10,**{'var':True})
    # mod = CombinedModelStation(env)
    mod.train(train)
    mod.save()
    mod.load()
    DI = DecisionIntervals(ud_train, mod, 0.5, 0.99) #alpha and beta
    WH = mod.get_all_factors(test)
    dis = 'P'
    # print( np.arange(int(WH.shape[0]/24)))
    # print(WH['Date/Heure'].tail(20))
    # print(WH.loc[73*24:(24*73)+23]['Heure'])
    # ################
    # for i in np.arange(int(WH.shape[0]/24)+1):
    #     # print("period "+str(i))
    #     print(i)
    #     # print(WH.loc[i*24:(24*i)+23])
    #     # print(int(WH.loc[0,'wday']))
    #     # print("********************")
    #     # a=0
    #     # while WH.loc[(i*24)+a]['Heure']!=0:
    #     #     a=a+1
    #     # print("Dia "+str(i))
    #     print(WH.loc[(i*24):(24*i)+23]['Date/Heure'])
    #     DI.compute_min_max_data_period(WT=WH.loc[(i*24):(24*i)+23], data=test, predict=True,save=True,day=i, **{'distrib': dis}) 
    #     # break
    ###############
    # i=25
    # DI.compute_min_max_data_period(WT=WH.loc[(i*24):(24*i)+23], data=validation, predict=True,save=True,day=i, **{'distrib': dis})

    # di = test.env.decision_intervals
    # DI.load_intervals(data,'C:/Users/Clara Martins/Documents/Doutorado/Pierre Code/Bixi_poly/resultats/stations_bixi_min_max_target.csv','')
    # r = 6 + 48 + 24 + 7 * 24 + 24 - 14
    # r = 0 #?
    # # valid = data.get_partialdata_per(0, 0.8)
    # env = Environment('Bixi', 'test')
    # data = Data(env)
    # WH = mod.get_factors(data) #eu comentei
   
    # WH = WH.iloc[0:10]
  
    # intervals  = DI.load_intervals(data, path, distrib)