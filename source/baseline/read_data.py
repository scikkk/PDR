import pandas as pd


def dataReader(dataList = None): 
    dataList = []
    with open('./lists/data_list.txt') as f:
            for s in f.readlines():
                if s[0] != '#':
                    dataList.append(s.strip('\n'))
    df_list = []
    for i in dataList:
        df_list.append(pd.read_csv(f"data/processed/{i}.csv"))
    data = pd.concat(df_list)
    return data
# print(result)

def testReader():
    df_list = []
    for i in range(10):
        if i+1 == 4:
            continue
        df = pd.read_csv(f"TestSet/processed/test{i+1}.csv")
        nan_point = 0
        for i in range(df.shape[0]):
            nan_point = i
            if pd.isna( df.loc[i][df.columns[-1]]) :
                # print(i)
                break
        df_list.append(df.head(nan_point))
    data = pd.concat(df_list)
    return data    
    
if __name__ == '__main__':
    print(testReader())