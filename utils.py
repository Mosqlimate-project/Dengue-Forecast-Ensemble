import os
import numpy as np
import pandas as pd
from epiweeks import Week
import scipy.stats as stats
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from sklearn.base import BaseEstimator, RegressorMixin

from scipy.optimize import minimize
from scipy.stats import lognorm

def replace_outliers_with_mean(arr, threshold=2):
    # Step 1: Calculate the Z-scores
    mean = np.mean(arr)
    std = np.std(arr)
    z_scores = (arr - mean) / std

    # Step 2: Identify outliers
    outliers = np.abs(z_scores) > threshold

    # Step 3: Replace outliers with the mean of non-outlier values
    mean_non_outliers = np.mean(arr[~outliers])
    arr[outliers] = mean_non_outliers

    return arr



def get_lognormal_pars(med, lwr, upr, alpha=0.95):
    def loss2(theta):
        tent_qs = lognorm.ppf([(1 - alpha) / 2, (1 + alpha) / 2], s=theta[1], scale=np.exp(theta[0]))
        if lwr == 0:
            attained_loss = abs(upr - tent_qs[1]) / upr
        else:
            attained_loss = abs(lwr - tent_qs[0]) / lwr + abs(upr - tent_qs[1]) / upr
        return attained_loss

    mustar = np.log(med)
    opt_result = minimize(loss2, x0=[mustar, 0.5], bounds=[(-5 * abs(mustar), 5 * abs(mustar)), (0, 10)], method='L-BFGS-B')
    
    return opt_result.x
    
def get_cases(data, state):
    '''
    Function to filter the cases data by state
    '''
    data_ = data.loc[data.uf == state]
    data_.loc[:, 'date']  = pd.to_datetime(data_.date)
    data_ = data_.rename(columns = {'date':'dates'})
    data_.set_index('dates', inplace = True)
    
    return data_

def get_epiweek(date):
    '''
    Capturing the epidemiological year and week from the date 
    '''
    epiweek = Week.fromdate(date)
    return (epiweek.year, epiweek.week)


class SprintModel(BaseEstimator, RegressorMixin):
    '''
    A class to fetch the predictions in mosqlimate API as a model.

    Attributes
    ---------------
    model_id: int. 
            ID of the model in the mosqlimate API
    state: str.
            TWO letter adm-1 code for Brazil. 
    mean: bool.
            If True the pred column is returned. Otherwise the prediction is sampled from 
            a {dist} distribution with mean = pred and std compute using lower and upper. 
    dist:str.
         Distribution to sample the predictions. Available options are `normal` and `poisson`.


    Methods
    --------------
    fit():
        It doesn't do anything, but you need this method to use stacking. 
    predict()
        Return the predictions based on a X that the first column refers to epidemiological year and the
        second one to the epidemiological week. 
    '''
    def __init__(self, model_id, state, mean = True, dist = 'normal'):
        # Initialize any parameters for the model
        self.model_id = model_id
        self.state = state
        self.mean = mean 
        self.dist = dist 

        df1 = pd.read_csv(f'./predictions/preds_{model_id}_{state}_2023.csv.gz')
        df1 = df1.dropna(axis =1)
        
        df2 = pd.read_csv(f'./predictions/preds_{model_id}_{state}_2024.csv.gz')
        df2 = df2.dropna(axis =1)
        df2 = df2.loc[df2.date <= '2024-06-02']

        for_path = f'./predictions/preds_{model_id}_{state}_2025.csv.gz'

        if os.path.exists(for_path):
            df3 = pd.read_csv(for_path)
            df3 = df3.dropna(axis =1)
            df = pd.concat([df1,df2,df3])
     
        else:
            df = pd.concat([df1,df2])

        df['epiweek'] = pd.to_datetime(df['date']).apply(get_epiweek)
        df['epi_year'] = df['epiweek'].apply(lambda x: x[0])
        df['epi_week'] = df['epiweek'].apply(lambda x: x[1])
        df.drop(['epiweek'], axis =1, inplace = True)
        df = df.reset_index(drop = True)

        self.df = df

    def fit(self, X, y):

        return self

    def predict(self, X, samples=1):
        '''
        Return the predictions based on an X that the first column refers to the epidemiological year and the
        second one to the epidemiological week. If mean == True, the predict returned refers to the `pred` columns 
        in the API. Otherwise the predict value will be sampled based on the normal or poisson distribution considering 
        a confidence interval of 90% for the predictions registered in the platform and using the columns `pred`, `lower`m
        and `upper` registered in the API. 
        '''
        df = self.df
        if self.mean:
            preds = []
            for i in np.arange(0, X.shape[0]):

                preds.append(df.loc[(df.epi_year == X[i,0]) & (df.epi_week == X[i,1])].pred.values[0]
                           )
                
        else:
            if self.dist == 'normal':
                confidence_level = 0.9
                z_value = stats.norm.ppf((1 + confidence_level) / 2)
                # calculate the standard deviation from interval size
                df['std_dev'] = (df.upper - df.lower)/(2*z_value)
                
                preds = []
                for i in np.arange(0, X.shape[0]):
                    df_ = df.loc[(df.epi_year == X[i,0]) & (df.epi_week == X[i,1])]
                    # upper = df_.upper.values[0]
                    # lower = df_.lower.values[0]
                    # mean = df_.pred.values[0]
                    mean = df_.pred.values[0]
                    std_dev = df_.std_dev.values[0]
                    
                    # std_dev = (upper - lower)/(2*z_value)
    
                    preds.append(np.array([0 if p < 0 else p for p in  np.random.normal(mean, std_dev, samples)]))

            elif self.dist == 'poisson':
                preds = []
                for i in np.arange(0, X.shape[0]):
                    df_ = df.loc[(df.epi_year == X[i,0]) & (df.epi_week == X[i,1])]
                    mean = df_.pred.values[0]
                    preds.append(np.random.poisson(mean, size = samples))

            elif self.dist == 'lognormal':
                preds = []
                for i in np.arange(0, X.shape[0]):
                    df_ = df.loc[(df.epi_year == X[i,0]) & (df.epi_week == X[i,1])]

                    if df_.pred.values[0]!= 0:
                        mean, std = get_lognormal_pars(df_.pred.values[0], 
                                                       df_.lower.values[0], 
                                                       df_.upper.values[0],
                                                       alpha=0.9)

                    
                        predicted_values = np.random.lognormal(mean, std, size = samples)
                        preds.append(replace_outliers_with_mean(predicted_values))
                            
                    else: 
                        preds.append(np.zeros(samples,))

        return np.array(preds) #[pred[0]]*len(X)


def get_data_slice(data, state, start_date = Week(2022, 40).startdate().strftime('%Y-%m-%d'), 
                                    end_date = '2024-06-02'):
    '''
    Function to get the samples to train the model for a specific state in a range of dates. 
    '''
        
    dates = pd.date_range(start= start_date,
              end= end_date,
              freq='W-SUN')

    df_ = get_cases(data, state)
    y = df_.loc[dates].casos.values

    X = pd.DataFrame()
    X['date'] = dates
    X['epiweek'] = pd.to_datetime(X['date']).apply(get_epiweek)
            
    # If you want separate columns for year and week
    X['epi_year'] = X['epiweek'].apply(lambda x: x[0])
    X['epi_week'] = X['epiweek'].apply(lambda x: x[1])
            
    X.drop(['epiweek', 'date'], axis =1, inplace = True)
    
    return X, y


def get_forecast_X(start_date = Week(2022, 40).startdate().strftime('%Y-%m-%d'), 
                                    end_date = '2024-06-02'):
    '''
    Function to get the samples to train the model for a specific state in a range of dates. 
    '''
        
    dates = pd.date_range(start= start_date,
              end= end_date,
              freq='W-SUN')


    X = pd.DataFrame()
    X['date'] = dates
    X['epiweek'] = pd.to_datetime(X['date']).apply(get_epiweek)
            
    # If you want separate columns for year and week
    X['epi_year'] = X['epiweek'].apply(lambda x: x[0])
    X['epi_week'] = X['epiweek'].apply(lambda x: x[1])
            
    X.drop(['epiweek', 'date'], axis =1, inplace = True)
    
    return X
    

def plot_coef(df_coef): 
    '''
    Function to plot the coeficients of the LASSO regression for each state
    '''
    fig = plt.figure(figsize=(15, 7.5))
    gs = gridspec.GridSpec(2, 6, figure=fig)
    
    # First row with three boxplots
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:6])
    
    # Second row with two boxplots
    ax4 = fig.add_subplot(gs[1, 1:3])
    ax5 = fig.add_subplot(gs[1, 3:5])
    
    # Plot the boxplots
    df_coef_ = df_coef.loc[df_coef.state == 'AM']
    
    ax1.bar(df_coef_.model_id, df_coef_.coef)
    
    ax1.set_title('AM')
    
    df_coef_ = df_coef.loc[df_coef.state == 'CE']
    
    ax2.bar(df_coef_.model_id, df_coef_.coef)
    
    ax2.set_title('CE')
    
    df_coef_ = df_coef.loc[df_coef.state == 'GO']
    
    ax3.bar(df_coef_.model_id, df_coef_.coef)
    
    ax3.set_title('GO')
    
    df_coef_ = df_coef.loc[df_coef.state == 'PR']
    
    ax4.bar(df_coef_.model_id, df_coef_.coef)
    
    ax4.set_title('PR')
    
    df_coef_ = df_coef.loc[df_coef.state == 'MG']
    
    ax5.bar(df_coef_.model_id, df_coef_.coef)
    
    ax5.set_title('MG')

    fig.suptitle('LASSO coeficients', fontsize = 14)
    
    
    for ax_ in [ax1,ax2,ax3,ax4,ax5]:
        ax_.set_ylabel('Coeficient')
        ax_.set_xlabel('Model id')
    
    # Adjust layout
    plt.tight_layout()
    
    plt.show()
    
