import pandas as pd
import os, shutil 

df = pd.read_csv('score_board_200.csv')
source_root = '/Users/seominjae/data/RAF-DB_aligned/valid'
target_root = '/Users/seominjae/data/RAF-DB_aligned/valid_balanced'
for i in range(7):
    temp = df[df['label'] == i]
    temp_target_root = os.path.join(target_root,str(i+1))
    os.makedirs(temp_target_root, exist_ok=True)
    for row in temp.iterrows():
        file_name = row[1]['ImageName']
        file_name = file_name.replace('test', 'valid')
        source_file = os.path.join(source_root,str(i+1),file_name)
        target_file = os.path.join(temp_target_root,file_name)
        shutil.copy(source_file,target_file)