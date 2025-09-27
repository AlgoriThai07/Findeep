import pandas as pd
import json

# Check consolidated.csv
# try:
#     df = pd.read_csv('consolidated.csv')
#     print('CSV Preview:')
#     print(df.head())
# except Exception as e:
#     print(f'Error reading consolidated.csv: {e}')

# Check filtered.json
try:
    with open('filtered_combined.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print('\nJSON Preview (first 5 rows):')
    for row in data[:5]:
        print(row)
except Exception as e:
        print(f'Error reading filtered_combined.json: {e}')

# # Check by_company.json
# try:
#     with open('by_company.json', 'r', encoding='utf-8') as f:
#         by_company = json.load(f)
#     print('\nBy Company Preview (first group):')
#     if by_company:
#         print(by_company[0])
#     else:
#         print('No company data found.')
# except Exception as e:
#     print(f'Error reading by_company.json: {e}')
