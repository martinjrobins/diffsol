import os
import pandas as pd
import matplotlib.pyplot as plt
import pathlib

# load data
base_dir = pathlib.Path(__file__).parent.resolve()
test = pd.read_csv(os.path.join(base_dir, f'{base_dir}/DailyDelhiClimateTest.csv'))
train = pd.read_csv(os.path.join(base_dir, f'{base_dir}/DailyDelhiClimateTrain.csv'))

# concatenate data
data = pd.concat([train, test])

# average measurements into months
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')
data = data.resample('ME').mean()

# change data to float indicating month since the start of the data
data['month'] = 12.0 * (data.index.year - data.index.year.min()) + data.index.month - 1.0
data = data.set_index('month')

# normalize data (meantemp,humidity,wind_speed,meanpressure)
data["meantemp"] = (data["meantemp"] - data["meantemp"].mean()) / data["meantemp"].std()
data["humidity"] = (data["humidity"] - data["humidity"].mean()) / data["humidity"].std()
data["wind_speed"] = (data["wind_speed"] - data["wind_speed"].mean()) / data["wind_speed"].std()
data["meanpressure"] = (data["meanpressure"] - data["meanpressure"].mean()) / data["meanpressure"].std()

# save data
data.to_csv(os.path.join(base_dir, f'{base_dir}/MonthlyDelhiClimate.csv'))

# plot data save to file
data.plot()
plt.savefig(os.path.join(base_dir, f'{base_dir}/monthly_delhi_climate.png'))
