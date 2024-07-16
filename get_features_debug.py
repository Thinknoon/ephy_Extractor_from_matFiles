import sys
import pandas as pd
from fp_extractor_visuialization_modified import extract_spike_features
from fp_extractor_visuialization_modified import get_cell_features
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

    filter_ = 10
    if (1/time[1]-time[0]) < 20e3:
        filter_ = (1/time[1]-time[0])/(1e3*2)-0.5
    df, df_related_features = extract_spike_features(time, current, voltage)
    Cell_Features = get_cell_features(df, df_related_features, time, current, voltage, curr_index_0)
    plt.close()
    Cell_Features.index = [cell_name]
    Cell_Features.to_csv(f'temp/{cell_name}.csv')

if __name__ == '__main__':
    all_passed_check = pd.read_csv('./all_cellCheck_metadata.csv',index_col=0)
    for i in ['20201113pLIPhs07c08']:
        merge_cell_features(i,all_passed_check)