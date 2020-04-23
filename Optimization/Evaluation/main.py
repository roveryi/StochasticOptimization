import os 
import pandas as pd 
import numpy as np
import pickle
from xgboost.sklearn import XGBRegressor
from scipy.stats import norm

def summary_fatality(percentage_miss_strength, sa):
    '''
    This function is developed for compute the total strength added by algorithm
    '''
    im_bin = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, float('inf')]
    retrof_ratio = {}
    prop = {}
    retrof_ds_ratio = {}
    retrof_ds_median = {}
    for i in range(1,9):
        temp = pd.DataFrame() 
        temp['Index'] = existing_prop['Index']
        temp['Xpeakstrength'] = np.divide(np.multiply(percentage_miss_strength[i-1], missing_strength['X Diff']) + peak_strength['Xpeakstrength'], peak_strength['Xpeakstrength'])
        temp['Zpeakstrength'] = np.divide(np.multiply(percentage_miss_strength[i-1], missing_strength['Z Diff']) + peak_strength['Zpeakstrength'], peak_strength['Zpeakstrength'])
        temp = pd.concat([temp,retrof_upper_limit]).min(level=0)
        retrof_ratio[i] = temp
        prop[i] = existing_prop.drop(columns=['Xpeakstrength', 'Zpeakstrength'])
        prop[i]['Xpeakstrength'], prop[i]['Zpeakstrength'] = retrof_ratio[i]['Xpeakstrength'], retrof_ratio[i]['Zpeakstrength']

        retrof_ds_ratio[i] = pd.DataFrame(data = prop[i]['Index'])
        retrof_ds_ratio[i]['complete_ratio'] = retrof_complete_ratio.predict(prop[i].iloc[:,1:])
        # Compute retrofitted buildings damage state medians 
        retrof_ds_median[i] = pd.DataFrame(data = np.multiply(retrof_ds_ratio[i]['complete_ratio'],existing_ds['Complete']))
        retrof_ds_median[i].insert(0,'Index', retrof_ds_ratio[i]['Index'])

    region_fatality = compute_fatality(sampled_building_info, total_fatality, retrof_ds_median, sa_simulated, sa, im_bin)
    return region_fatality

def compute_fatality(sampled_building_info, building_fatality, building_dsm, design_Sa, Sa_map, IM_bin):
    '''    
    This function is mainly used for computing regional earthquake losses given scenario
    sampled_building_info: assigned missing building information 
    building_loss: prototype archetype losses in each damage state 
    building_dsm: retrofitted prototype archetype damage state medians 
    Sa_map: ground shaking maps given scenario/event
    IM_bin: different retrofit strength is applied to different intensity bins
    '''

    buildings = pd.DataFrame(data = sampled_building_info['Index'])
    # Building collapse  at each location
    site_fatality = buildings.merge(total_fatality, how='left')

    # Compute probability of complete damage at each site given each shake map
    ds_prob_map = pd.DataFrame(np.zeros(Sa_map.shape))

    # Loop over all shake maps
    for k in range(8):
        for j in range(Sa_map.shape[1]):
            sub_Sa_map = Sa_map.loc[(design_Sa.iloc[:,0] < IM_bin[k+1]) & (design_Sa.iloc[:,0] >= IM_bin[k]), :].iloc[:,j]
            if sub_Sa_map.shape[0] == 0: 
                continue
            sub_building_map = buildings.loc[(design_Sa.iloc[:,0] < IM_bin[k+1]) & (design_Sa.iloc[:,0] >= IM_bin[k]),:]
            building_ds_median = sub_building_map.merge(building_dsm[k+1])
            ds_prob_map.loc[(design_Sa.iloc[:,0] < IM_bin[k+1]) & (design_Sa.iloc[:,0] >= IM_bin[k]), j] = norm.cdf(np.log(sub_Sa_map.astype('float64')), loc = np.log(building_ds_median.iloc[:,1]), scale = 0.6)
    collapse_prob = ds_prob_map.mean(axis = 1)

    portfolio_fatality = np.multiply(collapse_prob,site_fatality.iloc[:,1]).sum().sum()
    return portfolio_fatality

    
def ordinance_fatality_representative_events(sampled_building_info, building_fatality, building_dsm, Sa_map):
    '''    
    This function is mainly used for computing regional earthquake losses given scenario
    sampled_building_info: assigned missing building information 
    building_loss: prototype archetype losses in each damage state 
    building_dsm: retrofitted prototype archetype damage state medians 
    Sa_map: ground shaking maps given scenario/event
    '''
    buildings = pd.DataFrame(data = sampled_building_info['Index'])
    # Building collapse  at each location
    site_fatality = buildings.merge(building_fatality)

    # Building damage state median at each location 
    building_ds_median = buildings.merge(building_dsm, how = 'left')

    # Compute probability of complete damage at each site given each shake map
    ds_prob_map = pd.DataFrame(np.zeros(Sa_map.shape))
    # Loop over all shake maps
    for j in range(Sa_map.shape[1]):
        ds_prob_map.iloc[:,j] = norm.cdf(np.log(Sa_map.iloc[:,j]), loc = np.log(building_ds_median.iloc[:,1].astype('float')), scale = 0.6)
    collapse_prob = ds_prob_map.mean(axis = 1)

    portfolio_fatality = np.multiply(collapse_prob,site_fatality.iloc[:,1]).sum().sum()
    
    return portfolio_fatality

BaseDirectory = r'/u/project/hvburton/roveryi/Evaluation'

# BaseDirectory = r'C:\Users\User\Desktop\Optimization\Evaluation'
os.chdir(BaseDirectory)
retrof_complete_ratio = pickle.load(open('Retrof_Complete_Ratio.pkl','rb'))

sampled_building_info = pd.read_csv('buildinginfo.csv')
sa_simulated = pd.read_csv('Sa_simulated_real.csv', header = None)

# Read existing building damage state
existing_ds = pd.read_csv('building_ds.csv')

# Read existing building properties and pushover 
existing_prop = pd.read_csv('building_performance.csv')

# Read building configuration, which is mainly used for loss assessment 
building_config = pd.read_csv('building_configuration.csv')

# Read building fatality 
# total_fatality = pd.read_csv('fatality_rate.csv')
# Use complete loss as criteria: complete loss = building collapse loss + fatality loss ($2 million per fatality)
total_fatality = pd.read_csv('complete_loss.csv') 

# Read retrofit upper limit, which corresponding to NON SWOF building peak strength
retrof_upper_limit = pd.read_csv('UpperLimit.csv')

# Read missing strength
missing_strength = pd.read_csv('building_miss_strength.csv')

# Read existing building damage state medians
existing_ds = pd.read_csv('building_ds.csv')  

peak_strength = pd.DataFrame()
peak_strength['Index'] = existing_prop['Index']
peak_strength['Xpeakstrength'] = np.multiply(existing_prop['SeismicWeight'], existing_prop['Xpeakstrength'])
peak_strength['Zpeakstrength'] = np.multiply(existing_prop['SeismicWeight'], existing_prop['Zpeakstrength'])

ordinance_ds = pd.read_csv(r'ordinance_building_ds.csv')
ordinance_ds = ordinance_ds.drop(['Slight', 'Moderate', 'Severe'])

algorithm_best = [0.5373147823253948, 0.6776796686980564, 0.8108382470888009,0.8537486161975294, 0.8639264114170233, 0.7609222356570593,0.710229536829122, 0.5536388483053063]

seed = int(os.getenv('SGE_TASK_ID'))

for i in range(seed,seed+1):
    os.chdir(BaseDirectory + '/Events')
    Sa = pd.read_csv('Event%i.csv'%i, header = None)
    algorithm_loss = summary_fatality(algorithm_best, Sa)
    ordinance_loss = ordinance_fatality_representative_events(sampled_building_info, total_fatality, ordinance_ds, Sa)
    existing_loss = ordinance_fatality_representative_events(sampled_building_info, total_fatality, existing_ds.drop(['Slight', 'Moderate', 'Severe']), Sa)
    r = pd.DataFrame(data=[existing_loss,ordinance_loss,algorithm_loss])
    r.to_csv(BaseDirectory + '/Results/Event%i.csv'%i)
