import pandas as pd
import math
def output_result(out_path,in_path,time,pred_dx,pred_dy,pred_dir):
    input = pd.read_csv(in_path)
    t_id = 0
    for row in input[['Time (s)','Latitude (°)']].itertuples(index=True, name='Pandas'):
        # print(row[2])
        delta_x = 0  
        delta_y = 0 
        while(t_id < len(time)-1 and time[t_id] < row[1]):
            delta_x += pred_dx[t_id]
            delta_y += pred_dy[t_id]
            t_id += 1
        if math.isnan(row[2]):
            input['Direction (°)'][row[0]] = pred_dir[t_id]
            input['Latitude (°)'][row[0]] = input['Latitude (°)'][row[0]-1] + delta_x*1e-9
            input['Longitude (°)'][row[0]] = input['Longitude (°)'][row[0]-1] + delta_y*1e-9
    input.to_csv (out_path,index=False , encoding = "utf-8")