import pandas as pd

# The data size is huge so processing it will crash the computer hence we access and process it in chunks.
new_df = pd.read_json('Clothing_Shoes_and_Jewelry.json',lines = True,  chunksize = 1000000)

#Now we create 33 csv files to store these chunks in a sorted way as we want given below.
counter = 1
for chunk in new_df:
    new_df = pd.DataFrame(chunk[[ 'overall' , 'reviewText' , 'summary' ]])
    
    new_df1 = new_df[new_df['overall'] == 1].sample(4000)
    new_df2 = new_df[new_df['overall'] == 2].sample(4000)
    
    new_df3 = new_df[new_df['overall'] == 3].sample(8000)
    
    new_df4 = new_df[new_df['overall'] == 4].sample(4000)
    new_df5 = new_df[new_df['overall'] == 5].sample(4000)
    
    new_df_merge = pd.concat([new_df1 , new_df2 , new_df3 , new_df4 , new_df5] , axis=0 , ignore_index=True)
    new_df_merge.to_csv( str(counter)+'.csv' , index = False )
    counter += 1
    
#Merge all 33 csv files

#Search for all files in this directory or folder with the following name.
from glob import glob
filenames = glob('*.csv')
#['1.csv','2.csv',........,'33.csv']

#Now we merge all the files in variable dataframes.
dataframes = []
for f in filenames:
    dataframes.append(pd.read_csv(f))
#dataframes list will contain 33 dataframes
#[df1, df2,.....]

#Until now the files were 33 different arrays in variable dataframe. Now we merge them as single dataframe.
finaldf = pd.concat(dataframes, axis = 0, ignore_index = True)
#Now we save this dataframe as a .csv file for further preprocessing.
finaldf.to_csv('balanced_reviews.csv',index = False)