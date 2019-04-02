import pandas as pd
import numpy as np

df = pd.read_csv('ANLY501_WEEK11_DataForActivityHW_Fatalities.csv')
df = df.drop("State or Federal Program", axis = 1)
df = df.rename(index = str, columns = {"Years to Inspect Each Workplace Once":"State or Federal Program"})
us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
}
df["code"] = df["State"].map(us_state_abbrev)
for col in df.columns:
    df[col] = df[col].astype(str)
df['text'] = df['State'] + '<br>' +\
            "State Rank: " + df["State Rank"] + '<br>' +\
            "Fatalities Rate: " + df["Rate of Fatalities-2012"] + " Fatalities in 2012: " + df["Fatalities-2012"] + '<br>' +\
            "Injuries/Illnesses Rate: " + df["Number of Injuries/Illnesses 2012"] + " Injuries/Illnesses in 2012: " + df["Injuries/Illnesses 2012 Rate"]
a = "Number of Injuries/Illnesses 2012"
b = "Injuries/Illnesses 2012 Rate"
c = "Penalties FY 2013 (Average $)"
d = "Penalties FY 2013 (Rank)"
df = df.rename(index = str, columns = {a:b, b:a, c:d, d:c})
df = df.replace("nan", "", regex = True)
df.to_csv("clean_data.csv", index = False)
