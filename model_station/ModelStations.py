import time

import config
import numpy as np
import model_station.Prediction as pr
import model_station.Reduction as red
from modelUtils import maxi,mini
from preprocessing.Data import Data
import pandas as pd
import joblib


class ModelStations(object):
    def __init__(self, env, reduction_method, prediction_method, dim, **kwargs):
        self.hparam = {
            'var':False,
            'zero_prob':False,
            'norm': False,
            'hours': [],
            'load_red': False,
            'log': False,
            'mean': False,
            'decor': False,
            'red':{},
            'pred':{},
            'have_reduction':True, 
            'second_pred': {
            'pred': 'linear',
            },
        }
        #self.reduction = 1
        self.dim =  dim
        print("Dimension is "+str(dim))
        self.hparam.update(kwargs)
        self.env = env
        self.hours = self.hparam['hours']
        self.load_red = self.hparam['load_red']
        self.reduce = red.get_reduction(reduction_method)(env, dim=dim, log=self.hparam['log'], mean=self.hparam['mean'])
        
        self.reduce_inv = red.get_reduction('svdevol')(env, dim=self.dim,log=self.hparam['log'], mean=self.hparam['mean'])
        self.meanPredictor = pr.get_prediction(prediction_method)(dim=self.reduce.dim, reduction = self.hparam['have_reduction'], **self.hparam['pred'])
        # self.varPredictor = pr.get_prediction('linear')(dim=len(Data(env).get_stations_col(None)),
        #                                                 kwargs=kwargs)
        self.featurePCA = None
        self.sum = 0
        self.secondPredictor = pr.get_prediction(self.hparam['second_pred']['pred'])(dim=len(Data(env).get_stations_col(None)),
                                                           kwargs=self.hparam['second_pred'])
        self.name = reduction_method + ' ' + prediction_method
        # if (self.hparam['have_reduction']):
        #     print("Valor correto")
        # else: 
        #     print("Valor errado")
        
    def train(self,data:Data, t=False,stations=None,**kwargs):
        #print('Treino com redução')
        self.hparam.update(kwargs)
        data.hour = self.hours
        starttime = time.time()
        learn = data
        if self.load_red:
            
            try:
                self.reduce.load(add_path=self.env.system)
            except (FileNotFoundError, OSError):
                print('load failed, training redution')
                type(self.reduce).train(self.reduce, learn, **self.hparam['red'])
                type(self.reduce).save(self.reduce,add_path=self.env.system)
        else:
            #print("entrei "+str(type(self.reduce)))
            type(self.reduce).train(self.reduce, learn, **self.hparam['red'])
            type(self.reduce).save(self.reduce, add_path=self.env.system)
        x = self.reduce #Reduce Class
        learn2 = type(x).transform(x, learn) #the result of the reduction
        
        
        #print(learn2)
        if self.hparam['decor']:
            WH = self.get_decor_factors(learn)
        else: #here
            WH = self.get_factors(learn)  
        if self.hours != []:
            learn2 = learn2[np.max(self.hours):]
        
        # if not isinstance(stations, pd.Series):
        #     print("Treino com todas as estacoes")
        #     self.meanPredictor.train(WH, y=learn2, **self.hparam['pred'])
        # else:
        #     print("Treino com parte das estacoes")
        #     a =[]
        #     for i in stations:
        #         a.append('Start date '+str(i))
        #         a.append('End date '+str(i))
        #     self.meanPredictor.train(WH, y=learn.get_miniOD()[a].to_numpy(), **self.hparam['pred'])
        # print("aqui")
        # print(learn2)
        self.meanPredictor.train(WH, y=learn2, **self.hparam['pred'])
        
        if self.reduce.algo=='svdevol':
            n=1-(168*self.hparam['n_week'])/data.get_miniOD([]).shape[0]
            self.train_inv(data.get_partialdata_per(n, 1))
        if self.hparam['zero_prob']:
            self.train_zero_prob(learn,**self.hparam['pred'])
        if self.hparam['var']:
            self.train_variance(learn, **self.hparam['pred'])
        if t: print('train time', self.name, time.time() - starttime)
        return time.time() - starttime


    def train_wo_reduction(self, data:Data,station,t=False,**kwargs):
        print("Treino sem Redução")
        # print("Station : " + station[0])
        #self.hparam['have_reduction'] = False
        self.hparam.update(kwargs)
        data.hour = self.hours
        starttime = time.time()
        learn = data
        self.station = station
        if self.hparam['decor']:
            WH = self.get_decor_factors(learn)
        else: #here
            WH = self.get_factors(learn)  
        # print(station[0][-4:])
        
        # if config.capacity ==True:
        #     capi = ["capacity_6046","capacity_6333","capacity_6338","capacity_6335","capacity_6726","capacity_7004","capacity_7016"]
        #     for i in capi:
        #         if i != ("capacity_"+self.station[0][-4:]):
        #             WH.drop(columns=[i],inplace=True) 
        #print(np.size(WH))
        # self.meanPredictor.dim=1
        #print(list(WH))
        # print(WH['temp'])
        
        #self.meanPredictor.train(WH, y=learn.get_miniOD()[[station[0]]].to_numpy(), **self.hparam['pred'])
        self.preselect = Data(self.env).get_stations_col(None)
 
        self.meanPredictor.train(WH, y=learn.get_miniOD()[self.preselect].to_numpy(), **self.hparam['pred'])
     
        if self.reduce.algo=='svdevol':
            n=1-(168*self.hparam['n_week'])/data.get_miniOD([]).shape[0]
            self.train_inv(data.get_partialdata_per(n, 1))
        if self.hparam['zero_prob']:
            self.train_zero_prob(learn,**self.hparam['pred'])
        if self.hparam['var']:
            self.train_variance(learn, **self.hparam['pred'])
        if t: print('train time', self.name, time.time() - starttime)
        return time.time() - starttime

    def train_variance(self, learn, **kwargs):
        if isinstance(learn,Data):
            learn.hour=self.hours
        # train variance on old data
        if self.hparam['decor']:
            WH = self.get_decor_factors(learn)
        else:
            WH = self.get_factors(learn)
        res = self.predict(WH) # predict reduced data_trip
        WH = self.get_var_factors(learn) #weather

        e = (maxi(res, 0.01) - self.get_objectives(learn)) ** 2

        self.secondPredictor.train(WH, e, **kwargs)

    def train_zero_prob(self, learn, **kwargs):
        # WH = self.get_var_factors(learn)
        # res = self.predict(WH)
        # e = (maxi(res, 0.01) - self.get_objectives(learn)) ** 2
        obj = self.get_objectives(learn)==0
        WH = self.get_factors(learn)
        self.secondPredictor.train(WH, obj, **kwargs)

    # extract learning information
    def get_all_factors(self, learn):
        if isinstance(learn, Data):
            df = learn.get_miniOD(self.hours)
        else:
            df = learn
        col = [i for i in df.columns.values if 'date' not in i]
        # df.to_csv("data_preprocessing")
        return df[col]

    def get_factors_database(self,learn:(Data,pd.DataFrame)):
        df = learn.get_miniOD_database(self.env,self.hours)
        col = []
        for i in df.columns.values:
            for j in config.learning_var:
                if j==i or j == i[:-1] or j == i[:-2]:
                    col.append(i)
        col = np.unique(col)

        return df[col]

    def get_all_factors_database(self, learn):
        if isinstance(learn, Data):
            df = learn.get_miniOD_database(self.env,self.hours)
        else:
            df = learn
        col = [i for i in df.columns.values if 'date' not in i]
        return df[col]

    def get_decor_factors(self, learn:(Data,pd.DataFrame)):
        n=20
        df = self.get_factors(learn).to_numpy()
        if self.featurePCA is None:
            self.sum = df.var(axis=0)
            self.mean = df.mean(axis=0)
            df = (df-df.mean(axis=0))/self.sum
            from sklearn.decomposition.pca import PCA
            self.featurePCA = PCA(n).fit(df)
            joblib.dump(self.featurePCA,config.root_path+'featurePCA')
            np.save(config.root_path+'norm_features_var',self.sum   )
            np.save(config.root_path+'norm_features_mean',self.mean   )

        return pd.DataFrame(self.featurePCA.transform((df-df.mean(axis=0))/self.sum),columns=config.get_decor_var(n))
        # if isinstance(learn, Data):
        #     df = learn.get_miniOD(self.hours)
        # else:
        #     df = learn
        # col = config.get_decor_var(20)
        # return df[col]

    def get_factors(self, learn:(Data,pd.DataFrame)):
        if isinstance(learn, Data):
        	df = learn.get_miniOD(hours = self.hours)
        else:
            df = learn
        col = []
        
        # print("config.learning_var")
        # print(config.learning_var)
        #print(learn.env.name[:-1]+)
        # df.to_csv(learn.env.name[:-1]+'.csv')

        if isinstance(df,pd.DataFrame):
            for i in df.columns.values:
                for j in config.learning_var:
                    if j==i or j == i[:-1] or j == i[:-2]:
                        col.append(i)
        else:
            for i in df.index:
                for j in config.learning_var:
                    if j==i or j == i[:-1] or j == i[:-2]:
                        col.append(i)
        col = np.unique(col)
        # print("col")
        # print(col)
        # # print(learn.env.name[:-1])
        # print(df[col].head(5))
        # print(col)
        return df[col]
    def get_factors_database(self,learn:(Data,pd.DataFrame)):
        if isinstance(learn, Data):
            df = learn.get_miniOD_database(self.hours)
        else:
            df = learn
        col = []
        for i in df.columns.values:
            for j in config.learning_var:
                if j==i or j == i[:-1] or j == i[:-2]:
                    col.append(i)
        col = np.unique(col)
        return df[col]

    def get_var_factors(self, learn:[Data,pd.DataFrame]):

        if isinstance(learn, Data):
            df = learn.get_miniOD(self.hours)
        else:
            df = learn
        col = []
        if isinstance(df,pd.DataFrame):
            for i in df.columns.values:
                for j in config.learning_var:
                    if j == i or j == i[:-1] or j == i[:-2]:
                        col.append(i)
        else:
            for i in df.index:
                for j in config.learning_var:
                    if j == i or j == i[:-1] or j == i[:-2]:
                        col.append(i)
        col = np.unique(col)
        return df[col]
    def get_var_factors_database(self, learn:[Data,pd.DataFrame]):

        if isinstance(learn, Data):
            df = learn.get_miniOD_database(self.hours)
        else:
            df = learn
        col = []
        for i in df.columns.values:
            for j in config.learning_var:
                if j == i or j == i[:-1] or j == i[:-2]:
                    col.append(i)
        col = np.unique(col)
        return df[col]

    def get_objectives(self, learn:Data):
        miniOD = learn.get_miniOD(self.hours)
        return self.reduce.get_y(miniOD) #returna Data trip
        # return miniOD[learn.get_stations_col(2015)]

    def predict(self, x, t=False):

        starttime = time.time()
        if not isinstance(x,np.ndarray):
            if self.hparam['decor']:
                xt = self.get_decor_factors(x)
            else:
                xt = self.get_factors(x)
        else:
            xt=x

        pred1 = self.meanPredictor.predict(xt, )
        
        # if self.hparam['have_reduction']:
            #print("entrei" + str(type(self.reduce_inv)))
        # print(pred1)
        # print(np.shape(pred1))
        pred = type(self.reduce).inv_transform(self.reduce, pred1)

        # pd.DataFrame(pred).to_csv("prediction_"+str(self.dim)+".csv")
        #print()
            #pred = self.meanPredictor.predict(xt)
            
        # pred = type(self.reduce).inv_transform(self.reduce, pred)
        
        
        if t: print('prediction time', self.name, time.time() - starttime)
        return maxi(0.01, pred)

    def variance(self, x):

        
        xt = self.get_var_factors(x)
        pred = self.predict(x=x)
        var = maxi(self.secondPredictor.predict(xt), 0.01)
        var = maxi(pred - pred ** 2, var)
        return var

    def zero_prob(self, x):
        xt = self.get_factors(x)
        p = self.predict(x)
        res=maxi(self.secondPredictor.predict(xt),np.exp(-p))
        return mini(maxi(res, 0.01), 0.99)

    def save(self):
        self.reduce.hours=self.hours
        self.meanPredictor.hours = self.hours
        type(self.reduce).save(self.reduce, add_path=self.env.system)
        self.meanPredictor.save(add_path=self.reduce.algo + self.env.system)
        
        if self.hparam['decor']: #decor==false 
            joblib.dump(self.featurePCA,config.root_path+'featurePCA')
            np.save(config.root_path+'norm_features_var',self.sum)
            np.save(config.root_path+'norm_features_mean',self.mean)
        if self.hparam['var'] or self.hparam['zero_prob']: #False
            self.secondPredictor.save(add_path=self.reduce.algo + self.meanPredictor.algo + self.env.system)

    def load(self):
        self.reduce.load(add_path=self.env.system)
        self.meanPredictor.load(add_path=self.reduce.algo + self.env.system)
        if self.hparam['decor']: #False
            self.featurePCA = joblib.load(config.root_path+'featurePCA')
            self.sum = np.load(config.root_path+'norm_features_var.npy')
            self.mean = np.load(config.root_path+'norm_features_mean.npy')
        if self.hparam['var']:  #False
            self.secondPredictor.load(add_path=self.reduce.algo + self.meanPredictor.algo + self.env.system)
        if self.hparam['zero_prob']: #False
            self.secondPredictor.load(add_path=self.reduce.algo + self.meanPredictor.algo + self.env.system)
        try:
            self.hours = self.reduce.hours
        except AttributeError:
            try:
                self.hours = self.meanPredictor.hours
            except AttributeError:
                self.hours=[]

    def reset(self):
        self.reduce = red.get_reduction(self.reduce.algo)(self.env, dim=self.dim)
        self.meanPredictor = pr.get_prediction(self.meanPredictor.algo)(dim=self.reduce.dim,reduction = self.hparam['have_reduction'], **self.hparam['pred'])
        self.secondPredictor = pr.get_prediction(self.secondPredictor.algo)(
            dim=len(Data(self.env).get_stations_col(None)), kwargs=self.hparam)

    def load_or_train(self, data, **kwargs):
        self.hparam.update(kwargs)
        try:
            self.load()
        except (IOError,EOFError):
            self.train(data, **self.hparam['pred'])
            self.save()

    def get_y(self, x, since=None):
        return self.reduce.get_y(x, since)

    def train_inv(self, data):
        if self.hparam['decor']:
            x=self.get_decor_factors(data)
            x = x[x['Annee']==2018] #Mudar
        else:
            x = self.get_factors(learn=data)
            x = x[x['Annee']==2018]

        y = data.get_miniOD(hours=[])[data.get_miniOD(hours=[])['Annee']==2018] #All stations
        #y = data.get_miniOD(hours=[])
        #y = y.iloc[-60:-1]

        # if new_stations:
        #     y = y[self.get_new_stations()] # Select only new stations
        # else:
        y = y[data.get_stations_col()] #All stations
        self.reduce_inv.train_inv(x = self.meanPredictor.predict(x), y = y.to_numpy())

    def get_new_stations(self):
        #New stations of 2018
        return ['Start date 4000','Start date 4001','Start date 4002','End date 4000','End date 4001','End date 4002',
                'Start date 7082','Start date 7083','Start date 7084','End date 7082','End date 7083','End date 7084']

        #New sattions of 2019

        # return ['Start date 7085','Start date 7086','Start date 7087','Start date 7088','Start date 7089','Start date 7090','Start date 7091','Start date 7092','Start date 7093',
        # 'Start date 7094','Start date 7095','Start date 7096','Start date 7097','Start date 7098','Start date 7099','Start date 7100','Start date 7101','Start date 7102','Start date 7103',
        # 'Start date 7104','Start date 7105','Start date 7106','Start date 7107','Start date 7108','Start date 7109','Start date 7110','Start date 7111','Start date 7112','Start date 7113',
        # 'Start date 7114','Start date 7115','Start date 7116','Start date 7117','Start date 7118','Start date 7119','Start date 7120','Start date 7121','Start date 7122','Start date 7123',
        # 'Start date 7124','Start date 7125','Start date 7126','Start date 7127','Start date 7128','Start date 7129','Start date 7130','Start date 7131','Start date 7132','Start date 7133',
        # 'Start date 7134','Start date 7135','Start date 7136','Start date 7137','Start date 7138','Start date 7139','Start date 7140','Start date 7141','Start date 7142','Start date 7143',
        # 'Start date 7144','Start date 7145','Start date 7146','Start date 7147','Start date 7148','Start date 7149',
        # 'End date 7085','End date 7086','End date 7087','End date 7088','End date 7089','End date 7090','End date 7091','End date 7092','End date 7093','End date 7094','End date 7095',
        # 'End date 7096','End date 7097','End date 7098','End date 7099','End date 7100','End date 7101','End date 7102','End date 7103','End date 7104','End date 7105','End date 7106',
        # 'End date 7107','End date 7108','End date 7109','End date 7110','End date 7111','End date 7112','End date 7113','End date 7114','End date 7115','End date 7116','End date 7117',
        # 'End date 7118','End date 7119','End date 7120','End date 7121','End date 7122','End date 7123','End date 7124','End date 7125','End date 7126','End date 7127','End date 7128',
        # 'End date 7129','End date 7130','End date 7131','End date 7132','End date 7133','End date 7134','End date 7135','End date 7136','End date 7137','End date 7138','End date 7139' ,
        # 'End date 7140','End date 7141','End date 7142','End date 7143','End date 7144','End date 7145','End date 7146','End date 7147','End date 7148','End date 7149']

    def get_old_stations(self,x):
        st = x.get_stations_col()
        return np.setdiff1d(st, self.get_new_stations()).tolist()

    def get_group_old_stations(self):
        #New stations of 2018
        return ['Start date 4000','Start date 4001','Start date 4002','End date 4000','End date 4001','End date 4002',
                'Start date 7082','Start date 7083','Start date 7084','End date 7082','End date 7083','End date 7084']

        #New stations of 2019

        # return ['Start date 6162', 'End date 6162','Start date 6210','End date 6210','Start date 6323','End date 6323','Start date 6324','End date 6324',
        # 'Start date 6330','End date 6330','Start date 6331','End date 6331','Start date 6332','End date 6332','Start date 6333','End date 6333',
        # 'Start date 6334','End date 6334','Start date 6338','End date 6338','Start date 6410','End date 6410','Start date 6412','End date 6412',
        # 'Start date 6422','End date 6422','Start date 6710','End date 6710','Start date 6716','End date 6716','Start date 6726','End date 6726',
        # 'Start date 6736','End date 6736','Start date 6737','End date 6737','Start date 6917','End date 6917','Start date 7002','End date 7002',
        # 'Start date 7004','End date 7004','Start date 7005','End date 7005','Start date 7006','End date 7006','Start date 7063','End date 7063',
        # 'Start date 7064','End date 7064']

    def get_group_new_stations(self):
        return ['Start date 7090','End date 7090','Start date 7091','End date 7091','Start date 7092','End date 7092','Start date 7093','End date 7093',
        'Start date 7094','End date 7094','Start date 7095','End date 7095' ,'Start date 7096','End date 7096']