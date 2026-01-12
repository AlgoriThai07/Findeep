import pandas as pd
df = pd.read_json('filtered_combined.json')
df['Date'] = pd.to_datetime(df['Date'])
tesla = df[df['Company'] == 'Tesla'].sort_values('Date')
print(tesla[['Date', 'Revenues']].tail(10))
