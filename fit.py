import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from gwosc import datasets
import requests, os
from gwpy.timeseries import TimeSeries
# from gwosc.locate import get_urls

from pycbc.waveform import get_td_waveform



st.title('Fit the waveform to the data')

m1 = 10
m2 = 10
apx = 'SEOBNRv2'
# apx = 'IMRPhenomD'  #-- This one tapers by default




m1 = st.slider('Set mass 1', 10.0, 100.0, 20.0, step=0.1)
m2 = st.slider('Set mass 2', 10.0, 100.0, 20.0, step=0.1)
dist = st.slider('Set Distance in Mpc', 10, 4000, 2000)

hp, hc = get_td_waveform(approximant=apx,
                         mass1=m1,
                         mass2=m2,
                         spin1z=0,
                         delta_t=1.0/4096,
                         distance = dist,
                         f_lower=40)


#hp.taper()


fig = plt.figure()
plt.plot(hp.sample_times, hp, label=apx)
plt.ylabel('Strain')
plt.xlabel('Time (s)')
plt.legend()
#st.pyplot(fig)
plt.close()


# -- File download experiment (File Download Workaround)
# -- See http://awesome-streamlit.org
# -- https://raw.githubusercontent.com/MarcSkovMadsen/awesome-streamlit/master/gallery/file_download/file_download.py

data = {'Time':hp.sample_times, 'Strain':hp.data}
# When no file name is given, pandas returns the CSV as a string, nice.
#df = pd.DataFrame([data, hp.sample_times], columns=["Col1", 'col2'])
df = pd.DataFrame(data)
csv = df.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
href = f'<a href="data:file/csv;base64,{b64}">Download Waveform CSV File</a> (right-click and save as waveform.csv or waveform.txt)'
st.markdown(href, unsafe_allow_html=True)


# -- Get some data
import pylab
from pycbc.catalog import Merger
from pycbc.filter import resample_to_delta_t, highpass

@st.cache   #-- Magic command to cache data
def load_gw(t0, detector):
    strain = TimeSeries.fetch_open_data(detector, t0-18, t0+18, cache=False)
    return strain

#st.sidebar.markdown("## Select Data Time and Detector")

# -- Get list of events

t0 = datasets.event_gps('GW150914')

strain = load_gw(t0, 'H1')
strain = strain.crop(int(t0)-16, int(t0)+16)
plt.figure()
plt.plot(strain)
#st.pyplot()

import pycbc
data = pycbc.types.timeseries.TimeSeries(strain.value, delta_t=strain.dt.value)
data.start_time = strain.t0.value

plt.figure()
plt.plot(data.sample_times, data)
# st.pyplot()

#st.write(type(data.delta_t))
strain = highpass(data, 15.0)
conditioned = strain.crop(2, 2)
from pycbc.psd import interpolate, inverse_spectrum_truncation

psd = conditioned.psd(4)
psd = interpolate(psd, conditioned.delta_f)
psd = inverse_spectrum_truncation(psd, int(2 * conditioned.sample_rate),
                                  low_frequency_cutoff=15)

hp.resize(len(conditioned))
template = hp.cyclic_time_shift(hp.start_time)

from pycbc.filter import matched_filter

snr = matched_filter(template, conditioned,
                     psd=psd, low_frequency_cutoff=20)

snr = snr.crop(4 + 4, 4)

peak = abs(snr).numpy().argmax()
snrp = snr[peak]
time = snr.sample_times[peak]


from pycbc.filter import sigma
# The time, amplitude, and phase of the SNR peak tell us how to align
# our proposed signal with the data.

# Shift the template to the peak time
dt = time - conditioned.start_time
aligned = template.cyclic_time_shift(dt)

# scale the template so that it would have SNR 1 in this data
#aligned /= sigma(aligned, psd=psd, low_frequency_cutoff=20.0)

# Scale the template amplitude and phase to the peak value
aligned = (aligned.to_frequencyseries() * snrp).to_timeseries()

# -- Scale to get back to right distance
aligned /= np.abs(snrp)

aligned.start_time = conditioned.start_time


# We do it this way so that we can whiten both the template and the data
white_data = (conditioned.to_frequencyseries() / psd**0.5).to_timeseries()
white_template = (aligned.to_frequencyseries() / psd**0.5).to_timeseries()

white_data = white_data.highpass_fir(30., 512).lowpass_fir(300, 512)
white_template = white_template.highpass_fir(30, 512).lowpass_fir(300, 512)

# Select the time around the merger
white_data = white_data.time_slice(t0-.12, t0+.06)
white_template = white_template.time_slice(t0-.12, t0+.06)

plt.figure(figsize=[12, 3])

plt.plot(white_data.sample_times, white_data, label="Data")
plt.plot(white_template.sample_times, white_template, label="Template")
plt.ylim(-160, 160)
plt.legend()
st.pyplot()
