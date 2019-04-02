import plotly
plotly.tools.set_credentials_file(username='liyigao', api_key='6EmMOoSuy5cfO0f6PLyK')
##When you set-up plotly, you will have a .credentials file on your computer with this
##info as well

import plotly.plotly as py
import pandas as pd
import plotly.graph_objs as go

# Setup dataframe
##You should use YOUR path here. THis is location of MY data :)
dataFrame = pd.read_csv('ANLY501_WEEK11_DataForActivityHW_Fatalities.csv')

# Create a trace for a scatterplot
trace = go.Scatter(
	x = dataFrame['Number of Fatalities-2012'],
	y = dataFrame['Rate of Fatalities-2012'],
	mode = 'markers'
)

# Assign it to an iterable object named myData
myData = [trace]

# Add axes and title
myLayout = go.Layout(
	title = "Workplace Fatalities",
	xaxis=dict(
		title = 'Number of Fatalities-2012'
	),
	yaxis=dict(
		title = 'Rate of Fatalities-2012'
	)
)

# Setup figure
myFigure = go.Figure(data=myData, layout=myLayout)

# Create the scatterplot
#py.iplot(myFigure, filename='fatalities')
##NOTE: WHen you run the code, it may load a plotly error page
## Just refresh and choose to skip the login. 
py.plot(myFigure, filename='fatalities')

# Create boxplot
dataFrame = dataFrame.drop("State or Federal Program", axis = 1)
dataFrame = dataFrame.rename(index = str, columns = {"Years to Inspect Each Workplace Once":"State or Federal Program"})

inspectorState = dataFrame["Inspectors"][dataFrame["State or Federal Program"] == "State"]
inspectorFederal = dataFrame["Inspectors"][dataFrame["State or Federal Program"] == "Federal"]

trace0 = go.Box(y = inspectorState, name = "State")
trace1 = go.Box(y = inspectorFederal, name = "Federal")
data = [trace0, trace1]
myLayout1 = go.Layout(title = "Inspectors",
                      xaxis = dict(title = "State or Federal Program"),
                      yaxis = dict(title = "Number of Inspectors"))

myFigure1 = go.Figure(data = data, layout = myLayout1)
#py.iplot(myFigure1, filename = "boxplot")
py.plot(myFigure1, filename = "boxplot")

# Create map
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

for col in dataFrame.columns:
    dataFrame[col] = dataFrame[col].astype(str)

dataFrame['text'] = dataFrame['State'] + '<br>' +\
                    "State Rank: " + dataFrame["State Rank"] + '<br>' +\
                    "Fatalities Rate: " + dataFrame["Rate of Fatalities-2012"] + " Fatalities in 2012: " + dataFrame["Fatalities-2012"] + '<br>' +\
                    "Injuries/Illnesses Rate: " + dataFrame["Number of Injuries/Illnesses 2012"] + " Injuries/Illnesses in 2012: " + dataFrame["Injuries/Illnesses 2012 Rate"]
             
dataFrame["code"] = dataFrame["State"].map(us_state_abbrev)

data = [dict(type = "choropleth",
             autocolorscale = True,
             locations = dataFrame["code"],
             z = dataFrame["Rate of Fatalities-2012"],
             locationmode = "USA-states",
             text = dataFrame["text"],
             marker = dict(line = dict(color = "rgb(255,255,255)",
                                       width = 2)),
             colorbar = dict(title = "Rate(Fatalities)"))]

myLayout2 = go.Layout(title = "2012 Rate of Fatalities by State",
                 geo = dict(scope = 'usa',
                            projection = dict(type = "albers usa"),
                            showlakes = True,
                            lakecolor = "rgb(255,255,255)"))

myFigure2 = go.Figure(data = data, layout = myLayout2)
#py.iplot(myFigure2, filename = "map")
py.plot(myFigure2, filename = "map")
