'''
Author: scikkk 203536673@qq.com
Date: 2022-11-26 16:24:03
LastEditors: scikkk
LastEditTime: 2022-11-29 00:03:39
Description: file content
'''
import numpy as np
import pandas as pd




import numpy as np
from math import sin, cos, atan2, atan,isnan,pi
#  计算旋转角
def getYaw(accVals, magVals) ->float:
	roll = atan2(accVals[0],accVals[2])
	pitch = -atan(accVals[1]/(accVals[0]*sin(roll)+accVals[2]*cos(roll)))
	yaw = atan2(magVals[0]*sin(roll)*sin(pitch)+magVals[2]*cos(roll)*sin(pitch)+magVals[1]*cos(pitch), magVals[0]*cos(roll)-magVals[2]*sin(roll))
	return yaw


'''
6:
    ftemp =  180 + ftemp
10:
    ftemp =  360- ftemp	
0,3,8,9	
    ftemp = 270 + ftemp
'''
    

def AC_Azimuth( accVals,   magVals)->float:		
    ftemp = getYaw(accVals, magVals) * 180.0 / pi
    if(ftemp > 0):
        ftemp -= 360

    ftemp =  180+ftemp

    if(ftemp > 360.0):
        ftemp -= 360.0
    if(ftemp < 0):
        ftemp += 360.0
    return ftemp	

# data = pd.read_csv('./test_case0/processed/test_case0.csv')
data = pd.read_csv('./TestSet/processed/test7.csv')
acc = np.array(data[['acce_x','acce_y','acce_z']])
mag = np.array(data[['magnet_x','magnet_y','magnet_z']])
time = np.array(data['time'])
y_dir = np.array(data['Location_dir'])
# print(acc)
# print(mag)

cnt = 0
cnt_w = 0
for i in range(len(acc)):
    y = y_dir[i]
    if y == -1:
        continue
    if(isnan(y)):
        break
    cnt += 1
    pred_y = AC_Azimuth(acc[i],mag[i])
    bia = min(abs(pred_y - y), 360 - abs(pred_y - y))
    if  bia > 30 or bia < 0:
        cnt_w += 1
        print(time[i],'\t', pred_y,'\t',y,'\t',bia)
        
print(f"{cnt_w}/{cnt}={cnt_w/cnt*100}%")
'''
in_path = './test_case0/Location_input.csv'
input = pd.read_csv(in_path)
t_id = 0
for row in tqdm(input[['Time (s)','Direction (°)']].itertuples(index=True, name='Pandas')):
    # print(row[2])
    delta_x = 0  
    delta_y = 0 
    while(t_id < len(time)-1 and time[t_id] < row[1]):
        t_id += 1
    if isnan(row[2]):
        # print(time[t_id])
        # for i in range(-3,3):
        #     print(AC_Azimuth(acc[t_id+i],mag[t_id+i]))
        input['Direction (°)'][row[0]] = AC_Azimuth(acc[t_id],mag[t_id])
        input['Latitude (°)'][row[0]] = 0
        input['Longitude (°)'][row[0]] = 0
out_path = './test_case0/Location_output.csv'
input.to_csv (out_path,index=False , encoding = "utf-8")
'''
