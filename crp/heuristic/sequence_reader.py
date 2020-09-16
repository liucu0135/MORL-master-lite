import pandas as pd

def get_sequence(input_path, sheet=0):
    df=pd.read_excel(input_path,sheet)
    df=pd.DataFrame(df, columns=['CarModel','Color'])
    df['CarModel']=[ord(x)-64 for x in df['CarModel']]
    # df['Color']=[ord(x)-64 for x in df['Color']]
    return df.to_numpy()
    # return sequence['CarModel'].to_numpy(), sequence['Color'].to_numpy()

# df=get_sequence('distribute result0.xls')
# print(df[0]-df[1])




