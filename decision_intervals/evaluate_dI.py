import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from model_station.CombinedModelStation import CombinedModelStation
from decision_intervals import *
from preprocessing.preprocess import get_features
import time
#import concurrent.futures
from multiprocessing import Process
import pandas as pd 
from scipy.stats import norm, poisson, nbinom, skellam


def alert_lost_demand_both_inv(test_data_re,test_data_el,reb_type,subs,dis=''):
    """
    computes the worst case test value
    :param test_data: env and data
    :param distrib:
    :param path: path of the intervals
    :param arr_dep: chose test ('dep' 'arr' or '')
    :return: test values
    """
    
    DI = DecisionIntervals(test_data_re.env, test_data_el.env, None, None,0.5,0.5)

    # DI_re = DI.load_intervals_wo_transform(test_data_re, test_data_re.env.decision_intervals, 'RP')
    # DI_re.to_csv('inventory_interval_regular_'+name+'.csv')
    # # DI_re_max.to_csv('inventory_interval_regular_'+name+'_max.csv')
    # # DI_re_tar.to_csv('inventory_interval_regular_'+name+'_tar.csv')
    
    # DI_el = DI.load_intervals_wo_transform(test_data_el, test_data_el.env.decision_intervals, 'EP')
    # DI_el.to_csv('inventory_interval_eletric_'+name+'.csv')
    # DI_el_max.to_csv('inventory_interval_eletric_'+name+'_max.csv')
    # DI_el_tar.to_csv('inventory_interval_eletric_'+name+'_tar.csv')

    return DI.alert_lost_demand_both_inv(test_data_re= test_data_re, test_data_el= test_data_el, intervals_re= DI.load_intervals(test_data_re, test_data_re.env.decision_intervals, '_reg_'+dis),intervals_el= DI.load_intervals(test_data_el, test_data_el.env.decision_intervals, '_ele_'+dis), df_re=test_data_re.get_miniOD([]), df_el=test_data_el.get_miniOD([]),reb_type = reb_type,subs=subs)


def alert_lost_demand_bixi_inv(test_data_re,test_data_el,path,reb_type,subs):
    """
    computes the worst case test value
    :param test_data: env and data
    :param distrib:
    :param path: path of the intervals
    :param arr_dep: chose test ('dep' 'arr' or '')
    :return: test values
    """
    # DI_re = DecisionIntervals(test_data_re.env, None, 0.5, None)
    # DI_el = DecisionIntervals(test_data_el.env, None, 0.5, None)
    # di = 'C:/Users/Clara Martins/Documents/Doutorado/Trabalho de Pierre/Pierre Code/Bixi_poly/bixi_intervals/intervals_aug_jul_2019.csv'
    # dis = ''
    DI = DecisionIntervals(test_data_re.env, test_data_el.env, None, None,0.5,0.5)
    return DI.alert_lost_demand_bixi_inv(test_data_re= test_data_re, test_data_el= test_data_el, intervals= DI.load_intervals(test_data_re, path, ''), df_re=test_data_re.get_miniOD([]), df_el=test_data_el.get_miniOD([]),reb_type=reb_type,subs=subs)

def alert_lost_demand_hulot_inv(test_data_re,test_data_el,path,reb_type,subs):
    """
    computes the worst case test value
    :param test_data: env and data
    :param distrib:
    :param path: path of the intervals
    :param arr_dep: chose test ('dep' 'arr' or '')
    :return: test values
    """
    # DI_re = DecisionIntervals(test_data_re.env, None, 0.5, None)
    # DI_el = DecisionIntervals(test_data_el.env, None, 0.5, None)
    # di = 'C:/Users/Clara Martins/Documents/Doutorado/Trabalho de Pierre/Pierre Code/Bixi_poly/bixi_intervals/intervals_aug_jul_2019.csv'
    # dis = ''
    DI = DecisionIntervals(test_data_re.env, test_data_el.env, None, None,0.5,0.5)
    return DI.alert_lost_demand_hulot_inv(test_data_re= test_data_re, test_data_el= test_data_el, intervals= DI.load_intervals(test_data_re, path, ''), df_re=test_data_re.get_miniOD([]), df_el=test_data_el.get_miniOD([]),reb_type=reb_type,subs=subs)

def alert_lost_demand_hulot_inv_2(test_data_re,test_data_el,reb_type,subs):
    """
    computes the worst case test value
    :param test_data: env and data
    :param distrib:
    :param path: path of the intervals
    :param arr_dep: chose test ('dep' 'arr' or '')
    :return: test values
    """
    
    DI = DecisionIntervals(test_data_re.env, test_data_el.env, None, None,0.5,0.5)

    # DI_re = DI.load_intervals_wo_transform(test_data_re, test_data_re.env.decision_intervals, 'RP')
    # DI_re.to_csv('inventory_interval_regular_'+name+'.csv')
    # # DI_re_max.to_csv('inventory_interval_regular_'+name+'_max.csv')
    # # DI_re_tar.to_csv('inventory_interval_regular_'+name+'_tar.csv')
    
    # DI_el = DI.load_intervals_wo_transform(test_data_el, test_data_el.env.decision_intervals, 'EP')
    # DI_el.to_csv('inventory_interval_eletric_'+name+'.csv')
    # DI_el_max.to_csv('inventory_interval_eletric_'+name+'_max.csv')
    # DI_el_tar.to_csv('inventory_interval_eletric_'+name+'_tar.csv')

    return DI.alert_lost_demand_both_inv(test_data_re= test_data_re, test_data_el= test_data_el, intervals_re= DI.load_intervals(test_data_re, test_data_re.env.decision_intervals, 'HRP'),intervals_el= DI.load_intervals(test_data_el, test_data_el.env.decision_intervals, 'HEP'), df_re=test_data_re.get_miniOD([]), df_el=test_data_el.get_miniOD([]),reb_type = reb_type,subs=subs)

def new_alerts_lost_demand(test_data, distrib, path, WH,mod,algo,stations = None):
    """
    computes the worst case test value
    :param test_data: env and data
    :param distrib:
    :param path: path of the intervals
    :param arr_dep: chose test ('dep' 'arr' or '')
    :return: test values
    """
    DI = DecisionIntervals(test_data.env, None, 0.5, None)
    return DI.new_alerts_lost_demand(test_data= test_data,distrib= distrib,intervals= DI.load_intervals(test_data, path, distrib), df=test_data.get_miniOD([]) , WH=WH ,mod =mod ,algo=algo, stations = stations)




def tuning_alpha_beta(train_re, validation_re,train_el, validation_el,subs):
    
    mod1 = ModelStations(train_re.env, 'svd', 'gbt', dim=20, **{'var': True}) #checar aqui
    mod1.train(data=train_re, stations=None) 
    mod1.save()
    mod1.load()
    # WH1 = mod1.get_all_factors(validation_re)
    # print(WH1)

    #For electric bikes
    mod2 = ModelStations(train_el.env, 'svd', 'gbt', dim=20, **{'var': True}) #checar aqui
    mod2.train(data=train_el, stations=None) 
    mod2.save()
    mod2.load()
    # WH2 = mod2.get_all_factors(validation_re)
    # print(WH2)
    # return
    result = pd.DataFrame([])
    i=0
    rebalancing_cap = pd.read_csv('test_results/total_number_rebalancing_operations.csv')['rebalanced']
    # get_capacities(train_re =train_re,validation_re =validation_re,train_el = train_el,validation_el =validation_el)
    # for ar in [0.45,0.5]:
    # for br in [0.1]:
    for br in [0.1,0.3,0.5,0.7,0.9]:
            # for ae in [0.45,0.5]:
        # for be in [0.1,0.3,0.5,0.7,0.9]:
        
            DI = DecisionIntervals(env1=train_re.env,env2=train_el.env, mod1=mod1,mod2=mod2, beta1=br, beta2=br)
            WH = mod1.get_all_factors(validation_re)
            dis = 'R'
            # print(WH.columns)
            DI.compute_min_max_data(WH, validation_re,validation_el, True, **{'distrib': dis})
            # return
            # df1 = DI.load_intervals_wo_transform(validation_re, validation_re.env.decision_intervals, 'R')
            
            #create_inventory_intervals(alpha_reg=0.5,beta_reg=br,alpha_ele=0.5,beta_ele=be,train_re=train_re,validation_re=validation_re,train_el=train_el,validation_el=validation_el)
            
            results_code = alert_lost_demand_both_inv(test_data_re=validation_re,test_data_el=validation_el,reb_type = rebalancing_cap,subs=subs) 
            results_code.to_csv("tune_results/values_for_table.csv")
            # result.at[i,"alpha_re"] = ar
            result.at[i,"beta_re"] = br
            # result.at[i,"alpha_el"] = ae
            result.at[i,"beta_el"] = br
            result.at[i,"total_alert"] = results_code[0] + results_code[1] +results_code[2] +results_code[3]
            result.at[i,"alert_arr_regular"] = results_code[0]
            result.at[i,"alert_dep_regular"] = results_code[1]
            result.at[i,"alert_arr_electric"] = results_code[2]
            result.at[i,"alert_dep_electric"] = results_code[3]
            result.at[i,"total_lost"] = results_code[4] + results_code[5] + results_code[6] + results_code[7] 
            result.at[i,"lost_arr_regular"] = results_code[4]
            result.at[i,"lost_dep_regular"] = results_code[5]
            result.at[i,"lost_arr_electric"] = results_code[6]
            result.at[i,"lost_dep_electric"] = results_code[7]
            result.at[i,"nb_rebalanced"] = results_code[8]
            i=i+1
            print(result)
            
            # result.to_csv("tune_results/result_test_clara_daniel_model_max_random_miniversion.csv")

            # result.to_csv("tune_results/result_test_clara_new_model_plus_random.csv")
    return 


def get_inventory_intervals(train_re,test_re,train_el,test_el,model=3):
    mod1 = ModelStations(train_re.env, 'svd', 'gbt', dim=20, **{'var': True}) #checar aqui
    mod1.train(data=train_re, stations=None) 
    mod1.save()
    mod1.load()
    

    #For electric bikes
    mod2 = ModelStations(train_el.env, 'svd', 'gbt', dim=20, **{'var': True}) #checar aqui
    mod2.train(data=train_el, stations=None) 
    mod2.save()
    mod2.load()

    beta = [0.1,0.2,0.3,0.4]

    for b1 in beta:
        for b2 in beta:
            if (b1 in [0.1,0.2,0.3]) and (b2 in [0.1,0.2,0.3]):
                print('skip - > ('+str(b1)+','+str(b2)+")")
                continue
            # elif b1==0.4 and b2==0.1:
            #     print('skip - > ('+str(b1)+','+str(b2)+")")
            #     continue
            # else:
            #     print('doing - > ('+str(b1)+','+str(b2)+")")
            DI = DecisionIntervals(env1=train_re.env,env2=train_el.env, mod1=mod1,mod2=mod2, beta1=b1, beta2=b2)
            WH = mod1.get_all_factors(test_re)
            
            if model ==1:
                print("Beta reg = "+str(b1)+"| Beta ele = "+str(b2))
                DI.compute_min_max_data(WH, test_re,test_el,"model1", True, **{'distrib': ''})
            elif model ==2:
                print("Beta reg = "+str(b1)+"| Beta ele = "+str(b2))
                DI.compute_min_max_data(WH, test_re,test_el,"model2", True, **{'distrib': ''})
            elif model ==3:
                print("Beta reg = "+str(b1)+"| Beta ele = "+str(b2))
                DI.compute_min_max_data(WH, test_re,test_el,"model3", True, **{'distrib': ''})
            else:
                print("Error! Model not found!")

# def get_code_results(train_re,test_re,train_el,test_el, subs):
    
#     mod1 = ModelStations(train_re.env, 'svd', 'gbt', dim=20, **{'var': True}) #checar aqui
#     mod1.train(data=train_re, stations=None) 
#     mod1.save()
#     mod1.load()
#     WH1 = mod1.get_all_factors(test_re)
#     print(WH1)

#     #For electric bikes
#     mod2 = ModelStations(train_el.env, 'svd', 'gbt', dim=20, **{'var': True}) #checar aqui
#     mod2.train(data=train_el, stations=None) 
#     mod2.save()
#     mod2.load()
#     WH2 = mod2.get_all_factors(test_re)
#     print(WH2)
    
#     result = pd.DataFrame([])
#     i=0
#     rebalancing_cap = pd.read_csv('test_results_new/total_number_rebalancing_operations_subs_'+str(subs)+'.csv')['rebalanced']
#     # print(rebalancing_cap)
#     beta = [0.1,0.2,0.3,0.4]
#     result = pd.DataFrame([])

#     i=0
#     for b1 in beta:
#         for b2 in beta:
#             # DI = DecisionIntervals(env1=train_re.env,env2=train_el.env, mod1=mod1,mod2=mod2, beta1=b1, beta2=b2)
#             # WH = mod1.get_all_factors(test_re)
#             # DI.compute_min_max_data(WH, test_re,test_el, True, **{'distrib': ''})
#             results_code = alert_lost_demand_both_inv(test_data_re=test_re,test_data_el=test_el,reb_type = rebalancing_cap,subs=subs,dis='_max_'+str(b1)+'_'+str(b2)+'_') 
            
#             # create_inventory_intervals(alpha_reg=alpha_reg[i],beta_reg=beta_reg[i],alpha_ele=alpha_ele[i],beta_ele=beta_ele[i],train_re=train_re,validation_re=test_re,train_el=train_el,validation_el=test_el)
                        
#             # results_code = alert_lost_demand_both_inv(test_data_re=test_re,test_data_el=test_el,reb_type = reb_type, name= names[i]) 
#             # results_code.to_csv('test_results/result_test_version_'+reb_type+'_per_hour_station_'+names[i]+'.csv')
#             # result.at[i,"reb_type"] = 2
#             # result.at[i,"alpha_re"] = alpha_reg[i]
#             result.at[i,"beta_re"] = b1
#             # result.at[i,"alpha_el"] = alpha_ele[i]
#             result.at[i,"beta_el"] = b2
#             result.at[i,"total_alert"] = results_code[0] + results_code[1] +results_code[2] +results_code[3]
#             result.at[i,"alert_arr_regular"] = results_code[0]
#             result.at[i,"alert_dep_regular"] = results_code[1]
#             result.at[i,"alert_arr_electric"] = results_code[2]
#             result.at[i,"alert_dep_electric"] = results_code[3]
#             result.at[i,"alert_regular"] = results_code[0]+ results_code[1]
#             result.at[i,"alert_electric"] = results_code[2]+ results_code[3]
#             result.at[i,"total_lost"] = results_code[4] + results_code[5] + results_code[6] + results_code[7] 
#             result.at[i,"lost_arr_regular"] = results_code[4]
#             result.at[i,"lost_dep_regular"] = results_code[5]
#             result.at[i,"lost_arr_electric"] = results_code[6]
#             result.at[i,"lost_dep_electric"] = results_code[7]
#             result.at[i,"lost_regular"] = results_code[4]+ results_code[5]
#             result.at[i,"lost_electric"] = results_code[6]+ results_code[7]
#             result.at[i,"nb_rebalanced"] = results_code[8]
#             result.at[i,"empty_stations"] = results_code[9]
#             result.at[i,"full_stations"] = results_code[10]
#             i=i+1
#             print(result)
#             result.to_csv('test_results_new/result_test_ours_avg_new_model_mostunbalanced_subs_'+str(subs)+'.csv')

def get_code_results(train_re,test_re,train_el,test_el, ext,subs,b1,b2):
    # mod1 = ModelStations(train_re.env, 'svd', 'gbt', dim=20, **{'var': True}) #checar aqui
    # mod1.train(data=train_re, stations=None) 
    # mod1.save()
    # mod1.load()
    # WH1 = mod1.get_all_factors(test_re)
    # print(WH1)

    # #For electric bikes
    # mod2 = ModelStations(train_el.env, 'svd', 'gbt', dim=20, **{'var': True}) #checar aqui
    # mod2.train(data=train_el, stations=None) 
    # mod2.save()
    # mod2.load()
    # WH2 = mod2.get_all_factors(test_re)
    # print(WH2)

    np.set_printoptions(suppress=True)
    rebalancing_cap = pd.read_csv('test_results_new/total_number_rebalancing_operations_subs_'+str(subs)+'_round_regular.csv')['rebalanced']
    # print(rebalancing_cap)
    
    i=0
    print(ext)
    print("Sub = "+str(subs))

    results_code = alert_lost_demand_both_inv(test_data_re=test_re,test_data_el=test_el,reb_type =50,subs=subs,dis=str(b1)+'_'+str(b2)+'_'+str(ext)+'_plus010')
    # print(results_code)
    # results_code.to_csv('test_revision/inv_ours_subs_'+str(subs)+'_model_'+str(ext)+'.csv')


def get_code_results1(train_re,test_re,train_el,test_el, ext,subs):
    mod1 = ModelStations(train_re.env, 'svd', 'gbt', dim=20, **{'var': True}) #checar aqui
    mod1.train(data=train_re, stations=None) 
    mod1.save()
    mod1.load()
    WH1 = mod1.get_all_factors(test_re)
    print(WH1)

    #For electric bikes
    mod2 = ModelStations(train_el.env, 'svd', 'gbt', dim=20, **{'var': True}) #checar aqui
    mod2.train(data=train_el, stations=None) 
    mod2.save()
    mod2.load()
    WH2 = mod2.get_all_factors(test_re)
    print(WH2)

    np.set_printoptions(suppress=True)
    # rebalancing_cap = pd.read_csv('test_results_new/total_number_rebalancing_operations_subs_'+str(subs)+'_round_regular.csv')['rebalanced']
    # print(rebalancing_cap)
    # beta = [0.1,0.2,0.3]
    beta = [0.1,0.2,0.3,0.4]
    beta_new = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
    # beta=[0.0.1]
    result = pd.DataFrame([])
    
    i=0
    print(ext)
    print("Sub = "+str(subs))
    flag=0
    for b1 in beta_new:
        for b2 in beta_new:
    # for b1 in [0.2]:
    #     for b2 in [0.2]:
    #         # results_code = []
            # for x in np.arange(5):
            #     results_code.append(alert_lost_demand_both_inv(test_data_re=test_re,test_data_el=test_el,reb_type =rebalancing_cap,subs=subs,dis=str(b1)+'_'+str(b2)+'_'+str(ext)))
                
            #     # results_code = pd.concat([results_code,results_code1])
            # results_code = np.mean(results_code,axis=0)
            # flag = flag+1
            # if flag <= 29:
            #     print("Skip ("+str(b1)+","+str(b2)+")")
            #     continue

            # if (b1 in beta) and (b2 in beta):
            #     print("Skip ("+str(b1)+","+str(b2)+")")
            #     continue

            # print("Computing ("+str(b1)+","+str(b2)+")")
            # DI = DecisionIntervals(env1=train_re.env,env2=train_el.env, mod1=mod1,mod2=mod2, beta1=b1, beta2=b2)
            # WH = mod1.get_all_factors(test_re)
            # print(WH['Date/Heure'])
            # DI.compute_min_max_data(WH, test_re,test_el,ext, True, **{'distrib': ''})
            results_code = alert_lost_demand_both_inv(test_data_re=test_re,test_data_el=test_el,reb_type =50,subs=subs,dis=str(b1)+'_'+str(b2)+'_'+str(ext)+'_plus010')
            # print(results_code)
            # results_code.to_csv('test_revision/inv_ours_subs_'+str(subs)+'.csv')  
            
            
            result.at[i,"beta_re"] = b1
            result.at[i,"beta_el"] = b2
            result.at[i,"subs"] = subs
            result.at[i,"total_lost"] = results_code[7] 

            result.at[i,"percen_total_lost"] = results_code[6] 
            result.at[i,"percen_regular_lost"]  = results_code[4] 
            result.at[i,"percen_electric_lost"] = results_code[5] 
            result.at[i,"percen_lost_arr_regular"] = results_code[0]
            result.at[i,"percen_lost_dep_regular"] = results_code[1]
            result.at[i,"percen_lost_arr_electric"] = results_code[2]
            result.at[i,"percen_lost_dep_electric"] = results_code[3]
            result.at[i,"nb_rebalanced"] = results_code[8]
            result.at[i,"bikes_moved_reg"] = results_code[9]
            result.at[i,"bikes_moved_ele"] = results_code[10]
            result.at[i,"bikes_moved_totak"] = results_code[9]+results_code[10]
            print(result)
            i=i+1
            # print(result)
            result.to_csv('test_revision/test_our_'+ext+'_model_subs_'+str(subs)+'_plus010_new.csv')
  
    
def get_bixi_results(test_re,test_el):

    
    path = 'C:/Users/Clara/Documents/Doutorado/Trabalho de Pierre/Clara Code/Bixi3/bixi_intervals/bixi_intervall_2021-2022_new.csv'
    result = pd.DataFrame([])
    for subs in [0,1,2,3]:
        results_code = alert_lost_demand_bixi_inv(test_data_re=test_re,test_data_el=test_el,path=path,reb_type=50,subs=subs)
        # print(results_code)
        # results_code.to_csv('test_revision/inv_total_number_rebalancing_operations_subs_'+str(subs)+'_round_electric.csv')          
        # result.at[0,"total_alert"] = results_code[0] + results_code[1] 
        # result.at[0,"alert_arr"] = results_code[0]
        # result.at[0,"alert_dep"] = results_code[1]
        # result.at[0,"total_lost"] = results_code[2] + results_code[3] + results_code[4] + results_code[5] 
        # result.at[0,"lost_arr_regular"] = results_code[2]
        # result.at[0,"lost_dep_regular"] = results_code[3]
        # result.at[0,"lost_arr_electric"] = results_code[4]
        # result.at[0,"lost_dep_electric"] = results_code[5]
        # result.at[0,"nb_rebalanced"] = results_code[6]
        # result.at[0,"empty_stations"] = results_code[7]
        # result.at[0,"full_stations"] = results_code[8]
        # result.at[0,"total_alert"] = results_code[0] + results_code[1] 
        # result.at[0,"alert_arr"] = results_code[0]
        # result.at[0,"alert_dep"] = results_code[1]
        

        result.at[subs,"subs"] = subs
        result.at[subs,"total_lost"] = results_code[7] 

        result.at[subs,"percen_total_lost"] = results_code[6] 
        result.at[subs,"percen_regular_lost"]  = results_code[4] 
        result.at[subs,"percen_electric_lost"] = results_code[5] 
        result.at[subs,"percen_lost_arr_regular"] = results_code[0]
        result.at[subs,"percen_lost_dep_regular"] = results_code[1]
        result.at[subs,"percen_lost_arr_electric"] = results_code[2]
        result.at[subs,"percen_lost_dep_electric"] = results_code[3]
        
        
        # result.at[subs,"lost_arr_regular"] = results_code[4]
        # result.at[subs,"lost_dep_regular"] = results_code[5]
        # result.at[subs,"lost_arr_electric"] = results_code[6]
        # result.at[subs,"lost_dep_electric"] = results_code[7]
        result.at[subs,"nb_rebalanced"] = results_code[8]
        result.at[subs,"bikes_moved_reg"] = results_code[9]
        result.at[subs,"bikes_moved_ele"] = results_code[10]
        result.at[subs,"bikes_moved_totak"] = results_code[9]+results_code[10]
        print(result)
    result.to_csv('test_revision/per_result_test_bixi_new_model_subs_'+str(subs)+'_round_none.csv')          
    
def get_hulot_results(test_re,test_el,subs):
    
    rebalancing_cap = pd.read_csv('test_results_new/total_number_rebalancing_operations_subs_'+str(subs)+'.csv')['rebalanced']
    print(rebalancing_cap)
    beta = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    result = pd.DataFrame([])

    for i,b in enumerate(beta):
        path = 'C:/Users/Clara/Documents/Doutorado/Trabalho de Pierre/Clara Code/Bixi3/hulot_intervals/decision_interval_min_max_targetP_beta'+str(b)+'.csv'
        results_code = alert_lost_demand_hulot_inv(test_data_re=test_re,test_data_el=test_el,path=path,reb_type =rebalancing_cap,subs=subs)
        # results_code.to_csv("tune_results/values_for_table_hulot.csv")
        result.at[i,"beta"] = beta[i]
        # result.at[i,"alpha_el"] = alpha_ele[i]
        # result.at[i,"beta_el"] = beta_ele[i]
        result.at[i,"total_alert"] = results_code[0] + results_code[1] 
        result.at[i,"alert_arr"] = results_code[0]
        result.at[i,"alert_dep"] = results_code[1]
        result.at[i,"total_lost"] = results_code[2] + results_code[3] + results_code[4] + results_code[5] 
        result.at[i,"lost_arr_regular"] = results_code[2]
        result.at[i,"lost_dep_regular"] = results_code[3]
        result.at[i,"lost_arr_electric"] = results_code[4]
        result.at[i,"lost_dep_electric"] = results_code[5]
        # result.at[i,"lost_regular"] = results_code[2]+ results_code[5]
        # result.at[i,"lost_electric"] = results_code[6]+ results_code[7]
        result.at[i,"nb_rebalanced"] = results_code[6]
        result.at[i,"empty_stations"] = results_code[7]
        result.at[i,"full_stations"] = results_code[8]
        print(result)
    result.to_csv('test_results_new/result_test_hulot_new_model_random_subs_'+str(subs)+'.csv')

def get_hulot_results2(train_re,test_re,train_el,test_el, subs):

    mod1 = ModelStations(train_re.env, 'svd', 'gbt', dim=20, **{'var': True}) #checar aqui
    mod1.train(data=train_re, stations=None) 
    mod1.save()
    mod1.load()
    

    #For electric bikes
    mod2 = ModelStations(train_el.env, 'svd', 'gbt', dim=20, **{'var': True}) #checar aqui
    mod2.train(data=train_el, stations=None) 
    mod2.save()
    mod2.load()

    rebalancing_cap = pd.read_csv('test_results_new/total_number_rebalancing_operations_subs_'+str(subs)+'.csv')['rebalanced']
    print(rebalancing_cap)
    beta = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    result = pd.DataFrame([])

    i=0
    for b1 in beta:
        for b2 in beta:
            DI = DecisionIntervals(env1=train_re.env,env2=train_el.env, mod1=mod1,mod2=mod2, beta1=b1, beta2=b2)
            WH = mod1.get_all_factors(test_re)
            DI.compute_min_max_data2(WH, test_re,test_el, True, **{'distrib': ''})
            results_code = alert_lost_demand_both_inv(test_data_re=test_re,test_data_el=test_el,reb_type = rebalancing_cap,subs=subs,dis='H') 
            
            # create_inventory_intervals(alpha_reg=alpha_reg[i],beta_reg=beta_reg[i],alpha_ele=alpha_ele[i],beta_ele=beta_ele[i],train_re=train_re,validation_re=test_re,train_el=train_el,validation_el=test_el)
                        
            # results_code = alert_lost_demand_both_inv(test_data_re=test_re,test_data_el=test_el,reb_type = reb_type, name= names[i]) 
            # results_code.to_csv('test_results/result_test_version_'+reb_type+'_per_hour_station_'+names[i]+'.csv')
            # result.at[i,"reb_type"] = 2
            # result.at[i,"alpha_re"] = alpha_reg[i]
            result.at[i,"beta_re"] = b1
            # result.at[i,"alpha_el"] = alpha_ele[i]
            result.at[i,"beta_el"] = b2
            result.at[i,"total_alert"] = results_code[0] + results_code[1] +results_code[2] +results_code[3]
            result.at[i,"alert_arr_regular"] = results_code[0]
            result.at[i,"alert_dep_regular"] = results_code[1]
            result.at[i,"alert_arr_electric"] = results_code[2]
            result.at[i,"alert_dep_electric"] = results_code[3]
            result.at[i,"alert_regular"] = results_code[0]+ results_code[1]
            result.at[i,"alert_electric"] = results_code[2]+ results_code[3]
            result.at[i,"total_lost"] = results_code[4] + results_code[5] + results_code[6] + results_code[7] 
            result.at[i,"lost_arr_regular"] = results_code[4]
            result.at[i,"lost_dep_regular"] = results_code[5]
            result.at[i,"lost_arr_electric"] = results_code[6]
            result.at[i,"lost_dep_electric"] = results_code[7]
            result.at[i,"lost_regular"] = results_code[4]+ results_code[5]
            result.at[i,"lost_electric"] = results_code[6]+ results_code[7]
            result.at[i,"nb_rebalanced"] = results_code[8]
            result.at[i,"empty_stations"] = results_code[9]
            result.at[i,"full_stations"] = results_code[10]
            i=i+1
            print(result)
            result.to_csv('test_results_new/result_test_hulot2simulation_new_model_random_subs_'+str(subs)+'_teste1.csv')

def compute_scores_def(train_re:Data, validation_re:Data,test_re:Data,train_el:Data, validation_el:Data,test_el:Data):
    """
    compute the scores for several values of alpha and beta and saves to  di_scores.csv file
    :param d: test data
    :param train: train data
    :return: none
    """
  
    # Tune the hyperparameters
    # get_inventory_intervals(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,model=1)
    # get_inventory_intervals(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,model=2)
    # get_inventory_intervals(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,model=3)
    # get_code_results1(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,ext='model1',subs=0)
    # get_code_results1(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,ext='model1',subs=1)
    # get_code_results1(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,ext='model1',subs=2)
    # get_code_results1(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,ext='model1',subs=3)
   
    # get_code_results1(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,ext='model3',subs=0)
    # get_code_results1(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,ext='model3',subs=1)
    # get_code_results1(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,ext='model3',subs=2)
    # get_code_results1(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,ext='model3',subs=3)    # get_code_results(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,subs=1)
    # get_bixi_results(test_re=test_re,test_el=test_el)
    # func= "max"
    # get_code_results1(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,ext=func,subs=0)
    # get_code_results1(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,ext=func,subs=1)
    # get_code_results1(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,ext=func,subs=2)
    # get_code_results1(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,ext=func,subs=3)  
    # func= "avg"
    # get_code_results1(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,ext=func,subs=0)
    # get_code_results1(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,ext=func,subs=1)
    # get_code_results1(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,ext=func,subs=2)
    # get_code_results1(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,ext=func,subs=3)  
    

    func= "max"
    get_code_results(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,ext=func,subs=0,b1=0.4,b2=0.3)
    get_code_results(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,ext=func,subs=1,b1=0.4,b2=0.2)
    get_code_results(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,ext=func,subs=2,b1=0.3,b2=0.5)
    get_code_results(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,ext=func,subs=3,b1=0.4,b2=0.3)  
    # func= "avg"
    # get_code_results(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,ext=func,subs=0,b1=0.4,b2=0.5)
    # get_code_results(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,ext=func,subs=1,b1=0.4,b2=0.4)
    # get_code_results(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,ext=func,subs=2,b1=0.3,b2=0.5)
    # get_code_results(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,ext=func,subs=3,b1=0.4,b2=0.3)  
    
    # get_code_results1(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,ext=func,subs=0)
    # get_code_results1(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,ext=func,subs=0)
    # get_code_results1(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,ext=func,subs=3)

    

    # get_bixi_results(test_re=test_re,test_el=test_el)
    # 
    # get_bixi_results(test_re=test_re,test_el=test_el,subs=0)
    # get_bixi_results(test_re=test_re,test_el=test_el,subs=1)
    # get_bixi_results(test_re=test_re,test_el=test_el,subs=2)
    # get_bixi_results(test_re=test_re,test_el=test_el,subs=3)
    # get_hulot_results(test_re=test_re,test_el=test_el,subs=0)
    # get_hulot_results(test_re=test_re,test_el=test_el,subs=1)
    # get_hulot_results2(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,subs=0)
    # get_hulot_results2(train_re=train_re,test_re=test_re,train_el=train_el,test_el=test_el,subs=1)
    
    # tuning_alpha_beta(train_re=train_re, validation_re=validation_re,train_el=train_el, validation_el=validation_el)
    

    # tuning_alpha_beta(train_re=train_re, validation_re=test_re,train_el=train_el, validation_el=test_el,subs=0)
    # tuning_alpha_beta(train_re=train_re, validation_re=validation_re,train_el=train_el, validation_el=validation_el,reb_type = '2')
    # alpha,beta = new_tuning_alpha_beta(validation=validation, train=train,stations=stations,algo=2)
    # alpha,beta = new_tuning_alpha_beta(validation=validation, train=train,stations=stations,algo=5)

    
if __name__ == '__main__':
    from preprocessing.Environment import Environment
    from preprocessing.Data import Data
    ud = Environment('Bixi', 'train','regular')
    train_re = Data(ud)
    ud = Environment('Bixi', 'validation','regular')
    validation_re = Data(ud)
    ud = Environment('Bixi', 'test','regular')
    test_re = Data(ud)

    ud = Environment('Bixi', 'train','electric')
    train_el = Data(ud)
    ud = Environment('Bixi', 'validation','electric')
    validation_el = Data(ud)
    ud = Environment('Bixi', 'test','electric')
    test_el = Data(ud)

    # ud = Environment('Bixi', 'train','both')
    # train_bo = Data(ud)
    # ud = Environment('Bixi', 'validation','both')
    # validation_bo = Data(ud)
    compute_scores_def(train_re=train_re, validation_re=validation_re,test_re=test_re,train_el=train_el, validation_el=validation_el,test_el=test_el)
   