import os
import pandas as pd
from calculate_cell_features import extract_spike_features
from calculate_cell_features import get_cell_features
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
import glob
import scipy.io
import numpy as np

def merge_cell_features(cell_name,check_metadata):
    file_path = check_metadata.loc[cell_name,'file_path']
    temp_mat_load = scipy.io.loadmat(file_path)
    voltage = temp_mat_load['m_FP']['alldt'][0,0][0,0]
    current = temp_mat_load['m_FP']['StiStep'][0,0][0]
    time = np.arange(25000)/25000
    curr_index_0 = temp_mat_load['m_FP']['curr_index_0'][0,0][0,0]
    if len(current)!=voltage.shape[1]:
        print('Attention!!! current length is not equal to voltage')
        current = current[:voltage.shape[1]]
    current_step =  temp_mat_load['m_FP']['stepCurr'][0,0][0,0]
    filter_ = 6.
    if (1/time[1]-time[0]) < 20e3:
        filter_ = (1/time[1]-time[0])/(1e3*2)-0.5
    df, df_related_features = extract_spike_features(time, current, voltage,fil = filter_)
    Cell_Features = get_cell_features(df, df_related_features, time, current, voltage, curr_index_0,current_step=current_step)
    plt.close()
    Cell_Features.index = [cell_name]
    Cell_Features.to_csv(f'temp/{cell_name}.csv')
    df_related_features.to_csv(f'./single_cell_ephy_info/df_related_features/{cell_name}.csv')


if __name__ == '__main__':
    #清空temp文件夹，再存入数据
    files_to_remove = glob.glob('temp/*.csv')
    if len(files_to_remove)>0:
        print('cleaning temp dirs to save new data')
        for fileIndex in files_to_remove:
            os.remove('temp/*.csv')
    p = Pool(32)
    all_passed_check = pd.read_csv('./all_cellCheck_metadata.csv',index_col=0)
    all_passed_check = all_passed_check.loc[all_passed_check['pass_check']=='Y',]
    for i in all_passed_check.index:
        p.apply_async(merge_cell_features,args=(i,all_passed_check,))
    p.close()
    p.join()
    subset_features_files = glob.glob('./temp/*.csv')
    features = pd.DataFrame()
    for j in subset_features_files:
        current_features = pd.read_csv(j,index_col=0)
        features = pd.concat([features,current_features])
    features.to_csv(f'all_cell_features.csv')
