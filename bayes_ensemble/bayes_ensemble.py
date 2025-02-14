import numpy as np
import pandas as pd
from epiweeks import Week
from scipy.special import logsumexp
from scipy.optimize import minimize
from scipy.stats import lognorm, norm
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from scoringrules import crps_lognormal, crps_normal,  interval_score  

def invlogit(y):
    return 1 / (1 + np.exp(-y))

def alpha_01(alpha_inv):
    K = len(alpha_inv) + 1
    z = np.full(K-1, np.nan)  # Equivalent to rep(NA, K-1)
    alphas = np.zeros(K)      # Equivalent to rep(0, K)
    
    for k in range(K-1):
        z[k] = invlogit(alpha_inv[k] + np.log(1 / (K - k)))
        alphas[k] = (1 - np.sum(alphas[:k])) * z[k]
    
    alphas[K-1] = 1 - np.sum(alphas[:-1])
    return alphas

def pool_par_gauss(alpha, m, v):
    '''
    Function to get the output distribution of the normal distribution pool

    Parameters
    ----------
    alpha : array of float 
        Weigths assigned to each distribution in the pool.
    m : array of float
        mu parameter
    sd : array of float
        sd parameter
    Returns
    -------
    tuple
        The first one is the mu and the second one the sd parameter of the distribution.
    
    '''

    ws = alpha / v
    vstar = 1 / np.sum(ws)
    mstar = np.sum(ws * m) * vstar
    return mstar, np.sqrt(vstar)
    
def get_lognormal_pars(med, lwr, upr, alpha=0.90, fn_loss = 'median'):
    '''
    Function to represent a forecast, considering its known median, 
    lower and upper limit (considering a CI of alpha), as a normal logarithmic distribution
    
    Parameters
    ----------
    med :float 
        forecast median
    lwr :float
        forecast lower 
    upr  :float  
        forecast upper
    alpha:float
        alpha value used to define the upper and lower values. 

    Returns
    -------
    tuple
        The first one is the mu and the second one the sd parameter of the distribution.
    '''
    def loss_lower(theta):
        tent_qs = lognorm.ppf([(1 - alpha)/2, (1 + alpha)/2], s=theta[1], scale=np.exp(theta[0]))
        if lwr == 0:
            attained_loss = abs(upr - tent_qs[1]) / upr
        else:
            attained_loss = abs(lwr - tent_qs[0]) / lwr + abs(upr - tent_qs[1]) / upr
        return attained_loss
    
    def loss_median(theta):
        tent_qs = lognorm.ppf([0.5, (1 + alpha)/2], s=theta[1], scale=np.exp(theta[0]))
        if  med == 0:
            attained_loss = abs(upr - tent_qs[1]) / upr
        else:
            attained_loss = abs(med - tent_qs[0]) / med + abs(upr - tent_qs[1]) / upr
        return attained_loss
    
    if med == 0:
        mustar = np.log(0.1)
    else: 
        mustar = np.log(med)

    if fn_loss == 'median':
        result = minimize(loss_median, x0=[mustar, 0.5], bounds=[(-5 * abs(mustar), 5 * abs(mustar)), (0, 15)],method = "Nelder-mead", 
                          options={'xatol': 1e-6, 'fatol': 1e-6, 
                           'maxiter': 1000, 
                           'maxfev':1000})
    if fn_loss == 'lower':
            result = minimize(loss_lower, x0=[mustar, 0.5], bounds=[(-5 * abs(mustar), 5 * abs(mustar)), (0, 15)],method = "Nelder-mead",
                            options={'xatol': 1e-8, 'fatol': 1e-8, 
                           'maxiter': 5000,
                           'maxfev':5000})

    return result.x

def get_normal_pars(med, lwr, upr, alpha=0.90):
    '''
    Function to represent a forecast, considering its known median, 
    lower and upper limit (considering a CI of alpha), as a gaussian distribution
    
    Parameters
    ----------
    med :float 
        forecast median
    lwr :float
        forecast lower 
    upr  :float  
        forecast upper
    alpha:float
        alpha value used to define the upper and lower values. 

    Returns
    -------
    tuple
        The first one is the mu and the second one the sd parameter of the distribution.
    '''
    def loss2(theta):
        tent_qs = norm.ppf([(1 - alpha)/2, (1 + alpha)/2], loc=theta[0],  scale=theta[1])
        if lwr == 0:
            attained_loss = abs(upr - tent_qs[1]) / upr
        else:
            attained_loss = abs(lwr - tent_qs[0]) / lwr + abs(upr - tent_qs[1]) / upr
        return attained_loss

    mustar = med
    result = minimize(loss2, x0=[mustar, 2*mustar], bounds=[(-5 * abs(mustar), 5 * abs(mustar)), (0, 100000)],method = "Nelder-mead")
    return result.x

def get_df_log_pars(preds_, alpha = 0.9, fn_loss = 'median'):
    '''
    Function that takes a DataFrame with columns ordered as: date, pred, lower, upper, 
    model_id, and adds five additional columns: mu, sigma, fit_med, fit_lwr, and fit_upr. 
    The first two represent the mu and sigma parameters of the lognormal distribution, while the 
    others represent the estimated median, lower bound, and upper bound based on the parameters of the 
    lognormal distribution.
    
    Parameters
    ----------
    preds_ : pd.dataframe
        pandas DataFrame with columns date, pred, lower, upper and model_id
    alpha :float
        alpha used fo computed the CI
   
    Returns
    -------
    pd.dataframe 
        Dataframe with five additional columns: mu, sigma, fit_med, fit_lwr, and fit_upr.

    '''
    compute_pars_result = np.apply_along_axis(lambda row: get_lognormal_pars(med=row[1], lwr=row[2], upr=row[3], fn_loss = fn_loss), 1, preds_)
    
    par_df = pd.DataFrame(compute_pars_result, columns=["mu", "sigma"])
    
    # Combine the original preds and the computed parameters
    with_pars = pd.concat([preds_, par_df], axis=1)
    
    theo_preds_result = np.apply_along_axis(lambda row: lognorm.ppf([0.5, (1 - alpha) / 2, (1 + alpha) / 2], s=row[1], scale=np.exp(row[0])), 1, par_df)
        
    # Create a DataFrame for the theoretical predictions
    theo_pred_df = pd.DataFrame(theo_preds_result, columns=["fit_med", "fit_lwr", "fit_upr"])
    
    with_theo_preds = pd.concat([with_pars, theo_pred_df], axis=1)

    return with_theo_preds

def get_df_normal_pars(preds_, alpha = 0.9):
    '''
    Function that takes a DataFrame with columns ordered as: date, pred, lower, upper, 
    model_id, and adds five additional columns: mu, sigma, fit_med, fit_lwr, and fit_upr. 
    The first two represent the mu and sigma parameters of the normal distribution, while the 
    others represent the estimated median, lower bound, and upper bound based on the parameters of the 
    normal distribution.
    
    Parameters
    ----------
    preds_ : pd.dataframe
        pandas DataFrame with columns date, pred, lower, upper and model_id
    alpha :float
        alpha used fo computed the CI
   
    Returns
    -------
    pd.dataframe 
        Dataframe with five additional columns: mu, sigma, fit_med, fit_lwr, and fit_upr.

    '''

    compute_pars_result = np.apply_along_axis(lambda row: get_normal_pars(med=row[1], lwr=row[2], upr=row[3]), 1, preds_)
    
    par_df = pd.DataFrame(compute_pars_result, columns=["mu", "sigma"])
    
    # Combine the original preds and the computed parameters
    with_pars = pd.concat([preds_, par_df], axis=1)
    
    theo_preds_result = np.apply_along_axis(lambda row: norm.ppf([0.5, (1 - alpha) / 2, (1 + alpha) / 2], loc=row[0], scale=row[1]), 1, par_df)
        
    # Create a DataFrame for the theoretical predictions
    theo_pred_df = pd.DataFrame(theo_preds_result, columns=["fit_med", "fit_lwr", "fit_upr"])
    
    with_theo_preds = pd.concat([with_pars, theo_pred_df], axis=1)

    return with_theo_preds


def find_opt_LS_weights_all(obs, preds, order_models, dist = 'log_normal'):
    '''
    Function that generate the weights of the ensemble minimizing the Log Score between
    the cases and the ensemble distribution .

    Parameters
    -----------------
    obs: pd.dataframe 
        dataframe with columns date and casos;
    
    preds: pd.dataframe
       dataframe with columns date, mu, sigma, and model_id
    
    order: list 
        order of the different models in the model_id column 

    dist: str ['log_normal', 'normal']

        distribution used to represent the forecast 
    '''

    K = len(order_models)

    def loss(eta):

        ws = alpha_01(eta)

        score = 0
        for date in obs.date:
            preds_ = preds.loc[preds.date == date]
            preds_ = preds_.sort_values(by = 'model_id')
            preds_ = preds_.drop(['date'],axis =1).reset_index(drop = True)
            ms = preds_['mu']
            vs = preds_.sigma**2

            if len(ms) != K:
                raise ValueError("n_models and vs are not the same size!")
            if dist == 'log_normal':
                pool = pool_par_gauss(alpha=ws, m=ms, v=vs) 
                meanlog, sdlog = pool
                score = score -lognorm.logpdf(obs.loc[obs.date == date].casos, s=sdlog, scale=np.exp(meanlog))
            elif dist == 'normal':
                pool = pool_par_gauss(alpha=ws, m=ms, v=vs)
                mu, sigma = pool
                score = score -np.log(norm.pdf(obs.loc[obs.date == date].casos,loc =mu, scale = sigma))
        return score  

    initial_guess = np.full(K-1, 1/(K-1))
    opt_result = minimize(loss, initial_guess,  method = 'Nelder-mead')

    optimal_weights = alpha_01(opt_result.x)
    
    return {
        'weights': optimal_weights,
        'loss': opt_result.fun
    }

def get_epiweek(date):
    '''
    Capturing the epidemiological year and week from the date 
    '''
    epiweek = Week.fromdate(date)
    return (epiweek.year, epiweek.week)


def find_opt_CRPS_weights_all(obs, preds, order_models, dist = 'log_normal'):
    '''
    Function that generate the weights of the ensemble minimizing the CRPS between
    the cases and the ensemble distribution .

    Parameters
    -----------------
    obs: pd.dataframe 
        dataframe with columns date and casos;
    
    preds: pd.dataframe
       dataframe with columns date, mu, sigma, and model_id
    
    order: list 
        order of the different models in the model_id column 

    dist: str ['log_normal', 'normal']

        distribution used to represent the forecast 
    '''

    K = len(order_models)

    def loss(eta):

        ws = alpha_01(eta)

        score = 0
        for date in obs.date:
            preds_ = preds.loc[preds.date == date]
            preds_ = preds_.sort_values(by = 'model_id')
            preds_ = preds_.drop(['date'],axis =1).reset_index(drop = True)
            ms = preds_['mu']
          
            vs = preds_.sigma**2
   
            if len(vs) != K:
                print(date)
                raise ValueError("n_models and vs are not the same size!")

            if dist == 'log_normal':
                pool = pool_par_gauss(alpha=ws, m=ms, v=vs)
                meanlog, sdlog = pool
                score = score + crps_lognormal(observation = obs.loc[obs.date == date].casos, mulog = meanlog, sigmalog = sdlog)

            elif dist == 'normal':
                pool = pool_par_gauss(alpha=ws, m=ms, v=vs)
                mu, sigma = pool
                score = score + crps_normal(obs.loc[obs.date == date].casos, mu, sigma)
  
        return score  

    initial_guess = np.full(K-1, 1/(K-1))#
    opt_result = minimize(loss, initial_guess,  method = 'Nelder-mead')

    optimal_weights = alpha_01(opt_result.x)
    
    return {
        'weights': optimal_weights,
        'loss': opt_result.fun
    }


class Ensemble:
    """
    A class to compare to compute the weights and apply the ensemble of the models.

    Attributes
    ----------

    Methods 
    ---------

    compute_weights()
        Function to comp the weigths to generate the ensemble based on the observation data provided 

    apply_ensemble()
        Function to get the ensemble distribution given the weights computed using compute_weights
        or using a set of weigths provided by the user. 
    """
 
    def __init__(
        self,
        df: pd.DataFrame,
        order_models = list, 
        dist: str = 'log_normal',
        fn_loss:str = 'median'

    ):
        """
        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the columns `date`, `pred`, 'lower', 'upper', and 'model_id', .
        dist : str
            Distributoin used to parametrize de predictions in the datagrame.
        """

        try: 
            df = df[['date', 'pred', 'lower', 'upper', 'model_id']]

        except:
            raise ValueError(
                "The input dataframe must contain the columns: 'date', 'pred', 'lower', 'upper', 'model_id'"
            )
        

        if dist == 'normal':
            df = get_df_normal_pars(df, alpha = 0.9)
        elif dist == 'log_normal': 
            df = get_df_log_pars(df, alpha = 0.9, fn_loss=fn_loss)

        self.df = df 
        self.dist = dist 
        self.order_models = order_models
    
    def compute_weights(self, df_obs, metric = 'crps'): 
        '''
        Function to compute the weights of the ensemble distribution based on the data provided and the 
        metric selected 

        Parameters 
        ------------
        df_obs: pd.dataframe
            Dataframe with the column `date` and `casos`. 
        
        metric: str 
            ['crps', 'log_score']
        '''

        preds = self.df[['date', 'mu', 'sigma','model_id']]

        if metric == 'crps':

            weights = find_opt_CRPS_weights_all(obs=df_obs, preds= preds, order_models = self.order_models, dist = self.dist)

        elif metric == 'log_score': 

            weights = find_opt_LS_weights_all(obs=df_obs, preds= preds, order_models = self.order_models, dist = self.dist)

        self.weights = weights 

        return  weights
    
    def apply_ensemble(self, weights=None): 
        """
        Function to compute the ensemble distribution based on the weights computed using `compute_weights`
        if `weights` is None.  If weights is not None than the weights provided are used to compute the 
        ensemble distribution.  
        """
        try: 
            if weights == None: 
                weights = self.weights['weights']
        except:
            pass 
            
        preds = self.df

        df_for = pd.DataFrame()

        for d in preds.date.unique():
            preds_ = preds.loc[preds.date == d]
            pool = pool_par_gauss(alpha = weights, m = preds_.mu,
                    v = preds_.sigma**2)
        
            p = np.array([0.5, 0.05, 0.95])
            
            if self.dist == 'log_normal':
                quantiles = lognorm.ppf(p, s=pool[1], scale=np.exp(pool[0]))

            elif self.dist == 'normal':

                quantiles = norm.ppf(p, loc=pool[0], scale=pool[1])
            
            df_ = pd.DataFrame([quantiles], columns = ['pred', 'lower', 'upper'])
        
            df_['date'] = d
            
            df_for = pd.concat([df_for, df_], axis =0).reset_index(drop = True)

        df_for.date = pd.to_datetime(df_for.date)
        
        return df_for
    

def dlnorm_mix(omega, mu, sigma, weights, log=False):
    """
    Compute the PDF or log-PDF of a mixture of lognormal distributions for multiple `omega` values.
    
    Parameters:
        omega (array-like): Values where the mixture density is evaluated. Can be a single value or an array.
        mu (array-like): Means (in log-space) for the lognormal components.
        sigma (array-like): Standard deviations (in log-space) for the lognormal components.
        weights (array-like): Mixture weights (must sum to 1).
        log (bool): Whether to return the log-density.
        
    Returns:
        array: The mixture density or log-density evaluated at `omega`.
    """
    omega = np.atleast_1d(omega)  # Ensure `omega` is an array
    lw = np.log(weights)  # Log of weights
    K = len(mu)  # Number of components

    if len(sigma) != K or len(weights) != K:
        raise ValueError("mu, sigma, and weights must have the same length")

    # Compute log-PDFs for each component in a vectorized manner
    ldens = np.array([
        lognorm.logpdf(omega, s=sigma[i], scale=np.exp(mu[i]))
        for i in range(K)
    ]).T  # Transpose to align with omega dimensions

    # Combine using logsumexp for numerical stability
    if log:
        ans = logsumexp(lw + ldens, axis=1)
    else:
        ans = np.exp(logsumexp(lw + ldens, axis=1))

    return ans if ans.size > 1 else ans.item()  # Return scalar if input was scalar


def compute_ppf(mu, sigma, weights):
    """
    Compute the Percent-Point Function (PPF), which is the inverse of the CDF, 
    for a mixture of lognormal distributions.

    The function takes the parameters of a lognormal mixture (mean, standard deviation, and weights) 
    and returns the mixture values for the 5th, 50th, and 95th percentiles.

    Parameters:
        mu (array-like): Mean values (in log-space) for the lognormal components of the mixture.
        sigma (array-like): Standard deviation values (in log-space) for the lognormal components of the mixture.
        weights (array-like): Weights of each component in the lognormal mixture. These should sum to 1.

    Returns:
        np.ndarray: The x-values corresponding to the 5th, 50th, and 95th percentiles. 
    """
    # Step 1: Generate a range of x-values for the integration
    x = np.linspace(1e-6, 10**5, 10**5)
    
    # Step 2: Compute PDF values for the mixture of lognormal distributions
    pdf_values = dlnorm_mix(x, mu, sigma, weights, log=False)

    # Step 3: Normalize the PDF using the trapezoidal rule
    dx = np.diff(x)  # Compute spacing between consecutive x-values
    dx = np.append(dx, dx[-1])  # Ensure length matches the x array
    area = np.sum(pdf_values * dx)  # Approximate the area under the PDF
    pdf_values_normalized = pdf_values / area  # Normalize the PDF to ensure total area is 1

    # Step 4: Compute the CDF by integrating the normalized PDF using the trapezoidal rule
    cdf_values = cumulative_trapezoid(pdf_values_normalized, x, initial=0)

    # Step 5: Invert the CDF to obtain the PPF
    ppf_function = interp1d(cdf_values, x, bounds_error=False, fill_value="extrapolate")
    
    # Example: Given probabilities, find the corresponding x values
    p = [0.5, 0.05, 0.95]  # Example probabilities (e.g., 5th, 50th, and 95th percentiles)
    x_for_p = ppf_function(p)  # Get x-values corresponding to the probabilities

    return x_for_p


def crps_lognormal_mix(omega, mu, sigma, weights):

    K = len(mu)

    if len(sigma) != K:
        print('mu and sigma should be the same lenght')

    crpsdens = list(np.zeros(K))
    
    for i in np.arange(K):

        crpsdens[i] = crps_lognormal(observation = omega, mulog = mu[i], sigmalog = sigma[i])

    return np.dot(np.array(weights), np.array(crpsdens))#, crpsdens 


def find_opt_LS_weights_linear_mix(obs, preds, order_models):
    '''
    obs: dataframe com colunas date and casos;
    ms: dataframe com colunas: date, mu, sigma, model_id
    '''

    K = len(order_models)

    def loss(eta):
        ws = alpha_01(eta)

        score = 0
        for date in obs.date:
            #print(date)
            preds_ = preds.loc[preds.date == date]
            preds_ = preds_.sort_values(by = 'model_id')
            preds_ = preds_.drop(['date'],axis =1).reset_index(drop = True)
            
            #ms = preds_['mu']
            vs = preds_.sigma**2
   
            if len(vs) != K:
                print(date)
                raise ValueError("n_models and vs are not the same size!")
            
            score = score - dlnorm_mix(obs.loc[obs.date == date].casos,
                                       preds_["mu"].to_numpy(), preds_["sigma"].to_numpy(), weights =ws, log = True)

        return score  

    initial_guess = np.random.normal(size=K - 1)#np.full(K-1, 1/(K-1))#
    opt_result = minimize(loss, initial_guess,  method = 'Nelder-mead')

    optimal_weights = alpha_01(opt_result.x)
    
    return {
        'weights': optimal_weights,
        'loss': opt_result.fun
    }


def find_opt_CRPS_weights_linear_mix(obs, preds, order_models):
    '''
    obs: dataframe com colunas date and casos;
    ms: dataframe com colunas: date, mu, sigma, model_id
    '''

    K = len(order_models)

    def loss(eta):
        ws = alpha_01(eta)

        score = 0
        for date in obs.date:
            #print(date)
            preds_ = preds.loc[preds.date == date]
            preds_ = preds_.sort_values(by = 'model_id')
            preds_ = preds_.drop(['date'],axis =1).reset_index(drop = True)

            vs = preds_['sigma']**2
   
            if len(vs) != K:
                print(date)
                raise ValueError("n_models and vs are not the same size!")
    
            score = score + crps_lognormal_mix(obs.loc[obs.date == date].casos,
                                       preds_["mu"].to_numpy(), preds_["sigma"].to_numpy(), weights =ws)

        return score  

    initial_guess = np.full(K-1, 1/(K-1))#
    opt_result = minimize(loss, initial_guess,  method = 'Nelder-mead')

    optimal_weights = alpha_01(opt_result.x)
    
    return {
        'weights': optimal_weights,
        'loss': opt_result.fun
    }


class Ensemble_linear:
    """
    A class to compare to compute the weights and apply the ensemble of the models
    assuming a linear mix of log normal distributions.

    Attributes
    ----------

    Methods 
    ---------

    compute_weights()
        Function to comp the weigths to generate the ensemble based on the observation data provided 

    apply_ensemble()
        Function to get the ensemble distribution given the weights computed using compute_weights
        or using a set of weigths provided by the user. 
    """
 
    def __init__(
        self,
        df: pd.DataFrame,
        order_models: list,
        fn_loss: str ,

    ):
        """
        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with the columns `date`, `pred`, 'lower', 'upper', and 'model_id', .
        dist : str
            Distributoin used to parametrize de predictions in the datagrame.
        """

        try: 
            df = df[['date', 'pred', 'lower', 'upper', 'model_id']]

        except:
            raise ValueError(
                "The input dataframe must contain the columns: 'date', 'pred', 'lower', 'upper', 'model_id'"
            )
        
        df = get_df_log_pars(df, alpha = 0.9, fn_loss=fn_loss)

        self.df = df 
        self.order_models = order_models
    
    def compute_weights(self, df_obs, metric = 'crps'): 
        '''
        Function to compute the weights of the ensemble distribution based on the data provided and the 
        metric selected 

        Parameters 
        ------------
        df_obs: pd.dataframe
            Dataframe with the column `date` and `casos`. 
        
        metric: str 
            ['crps', 'log_score']
        '''

        preds = self.df[['date', 'mu', 'sigma','model_id']]

        if metric == 'crps':

            weights = find_opt_CRPS_weights_linear_mix(obs=df_obs, preds= preds, order_models = self.order_models)

        elif metric == 'log_score': 

            weights = find_opt_LS_weights_linear_mix(obs=df_obs, preds= preds, order_models = self.order_models)

        self.weights = weights 

        return  weights
    
    def apply_ensemble(self, weights=None): 
        """
        Function to compute the ensemble distribution based on the weights computed using `compute_weights`
        if `weights` is None.  If weights is not None than the weights provided are used to compute the 
        ensemble distribution.  
        """
        try: 
            if weights == None: 
                weights = self.weights['weights']
        except:
            pass 
            
        preds = self.df

        df_for = pd.DataFrame()

        for d in preds.date.unique():
            preds_ = preds.loc[preds.date == d]

            quantiles = compute_ppf(mu = preds_['mu'].values, sigma = preds_['sigma'].values,
                                    weights = weights)
            
            df_ = pd.DataFrame([quantiles], columns = ['pred', 'lower', 'upper'])
        
            df_['date'] = d
            
            df_for = pd.concat([df_for, df_], axis =0).reset_index(drop = True)

        df_for.date = pd.to_datetime(df_for.date)
        
        return df_for
    
class Scorer:
    """
    A class to compare the score of the models.
    """

    def __init__(
        self,
        df_true: pd.DataFrame,
        pred: pd.DataFrame,
        confidence_level: float = 0.90,
    ):
        """
        Parameters
        ----------
        df_true: pd.DataFrame
            DataFrame with the columns `date` and `casos`.
        ids : list[int]
            List of the predictions ids that it will be compared.
        pred: pd.DataFrame
            Pandas Dataframe already in the format accepted by the platform
            that will be computed the score.
        confidence_level: float.
            The confidence level of the predictions of the columns upper and lower.
        """

        # input validation data
        cols_df_true = ["date", "casos"]

        if not set(cols_df_true).issubset(set(list(df_true.columns))):
            raise ValueError(
                "Missing required keys in the df_true:" f"{set(cols_df_true).difference(set(list(df_true.columns)))}"
            )

        df_true.date = pd.to_datetime(df_true.date)
        df_true = df_true.sort_values(by = 'date')
        # Ensure all the dates has the same lenght
        min_dates = [min(df_true.date)]
        max_dates = [max(df_true.date)]

        cols_preds = ["date", "lower", "pred", "upper"]
        if not set(cols_preds).issubset(set(list(pred.columns))):
            raise ValueError(
                    "Missing required keys in the pred:" f"{set(cols_preds).difference(set(list(pred.columns)))}"
            )

        pred.date = pd.to_datetime(pred.date)
        pred = pred.sort_values(by = 'date')

        pred = get_df_log_pars(pred[['date', 'pred', 'lower', 'upper']].reset_index(drop=True), fn_loss = 'median')

        min_date = min(pred.date)
        max_date = max(pred.date)

        # updating the dates interval
        df_true = df_true.loc[(df_true.date >= min_date) & (df_true.date <= max_date)]
        df_true = df_true.sort_values(by="date")
        df_true.reset_index(drop=True, inplace=True)

        self.df_true = df_true
        self.df_pred = pred
        self.confidence_level = confidence_level

    @property
    def crps(
        self,
    ):
        """
        tuple of dict: Dict where the keys are the id of the models or `pred`
        when a dataframe of predictions is provided by the user,
        and the values of the dict are the scores computed.

        The first dict contains the CRPS score computed for every predicted
        point, and the second one contains the mean values of the CRPS score
        for all the points.

        The CRPS computed assumes a normal distribution.
        """

        df_true = self.df_true
        df_pred = self.df_pred

        score = crps_lognormal(df_true.casos.values, df_pred.mu, df_pred.sigma)

        return score, np.mean(score)

    @property
    def log_score(
        self,
    ):
        """
        tuple of dict: Dict where the keys are the id of the models or `pred`
        when a dataframe of predictions is provided by the user, and the values
        of the dict are the scores computed.

        The first dict contains the log score computed for every predicted
        point, and the second one contains the mean values of the log score
        for all the points.

        The log score computed assumes a normal distribution.
        """

        df_true = self.df_true
        df_pred = self.df_pred

        score =  lognorm.logpdf(df_true.casos.values, s=df_pred.sigma.values, scale=np.exp(df_pred.mu.values))

        # logs_lognormal(df_true.casos.values, df_pred.mu, df_pred.sigma)

        score = np.maximum(score, np.repeat(-100, len(score)))

        return score, np.mean(score)

    @property
    def interval_score(
        self,
    ):
        """
        tuple of dict: Dict where the keys are the id of the models or `pred`
        when a dataframe of predictions is provided by the user,
        and the values of the dict are the scores computed.

        The first dict contains the interval score computed for every predicted
        point, and the second one contains the mean values of the interval score
        for all the points.
        """

        
        df_true = self.df_true
        df_pred = self.df_pred

        score = interval_score(
                                df_true.casos.values,
                                df_pred.lower,
                                df_pred.upper,
                                alpha = 0.1)
                                       
        return score, np.mean(score)