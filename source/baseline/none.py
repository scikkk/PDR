import pandas as pd
import math
last = 0
input = pd.read_csv("./test_case0/Location_input.csv")
for row in input[['Time (s)','Latitude (°)']].itertuples(index=True, name='Pandas'):
    # print(row[2])
    delta_x = 0  
    delta_y = 0
    if not math.isnan(row[2]):
        last = row[0]
    else:
        input['Direction (°)'][row[0]] = input['Direction (°)'][last]
        input['Latitude (°)'][row[0]] = input['Latitude (°)'][last]
        input['Longitude (°)'][row[0]] = input['Longitude (°)'][last]
input.to_csv ("./test_case0/none_Location_output.csv",index=False , encoding = "utf-8")