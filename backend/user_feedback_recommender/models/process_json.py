import pandas as pd


df = pd.read_json('restaurant_pages.json')

df.at[162, 'phone number'] = df.at[162, 'phone']


df.drop(columns=[
    'nearby','location', 'phone', "price_range",                  
   "best nearby",
   "special diets",                
   "popular dishes",               
   "popular restaurant categories",
   "best nearby restaurants",      
   "best nearby attractions",      
   "popular types of food",        
   "hours",                        
   "location",                     
   "nearby hotels",                
   "nearby restaurants",           
   "nearby attractions" 
    ], inplace=True)

df.loc[df['service'].apply(lambda x: isinstance(x, list)), 'features'] = df.apply(
    lambda row: row['features'] + row['service'] if isinstance(row['service'], list) else row['features'], axis=1
)


df.loc[df['atmosphere'].apply(lambda x: isinstance(x, list)), 'features'] = df.apply(
    lambda row: row['features'] + row['atmosphere'] if isinstance(row['atmosphere'], list) else row['features'], axis=1
)

# Drop 'service' column (in-place)
df.drop(columns=['food', 'service', 'value', 'atmosphere'], inplace=True)


df.reset_index(drop=True, inplace=True) 

# Save DataFrame as JSON
df.to_json('restaurant_pages.json', orient="records", indent=4)