from tslearn.utils import load_time_series_txt
import numpy
import matplotlib.pyplot as plt
import pandas as pd
from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from datetime import datetime
from tslearn import metrics

if __name__ == '__main__':
    dataframe = pd.read_csv('power.csv', index_col='Timestamp',
                            parse_dates=['Timestamp'])

    # dataframe['Power(kW)'].asfreq('M').plot()
    # plt.title('Power(kW) generated')
    # plt.show()

    dataset = dataframe['Power(kW)'].to_numpy()
    formatted_dataset = to_time_series_dataset(dataframe['Power(kW)'])
    n_ts, sz, d = formatted_dataset.shape
    scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)
    scaled_dataset = scaler.fit_transform(formatted_dataset)
    envelope_down, envelope_up = metrics.lb_envelope(scaled_dataset[0], radius=3)

    dataframe['Power(kW)'] = scaled_dataset[0]
    dataframe['UpperEnvelope'] = envelope_up[:, 0]
    dataframe['LowerEnvelope'] = envelope_down[:, 0]

    dataframe['Power(kW)'].asfreq('M').plot()
    dataframe['UpperEnvelope'].asfreq('M').plot()
    dataframe['LowerEnvelope'].asfreq('M').plot()
    plt.title('Envelope with r=3 around Power Generated(kW)')
    plt.savefig('plot.png')
    plt.show()
