import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64

import requests, os
# from gwpy.timeseries import TimeSeries
# from gwosc.locate import get_urls

from pycbc.waveform import get_td_waveform



st.title('Plot Waveform')

m1 = 10
m2 = 10
apx = 'SEOBNRv2'
# apx = 'IMRPhenomD'  #-- This one tapers by default




m1 = st.slider('Set mass 1', 1.0, 100.0, 10.0, step=0.1)
m2 = st.slider('Set mass 2', 1.0, 100.0, 10.0, step=0.1)

hp, hc = get_td_waveform(approximant=apx,
                         mass1=m1,
                         mass2=m2,
                         spin1z=0,
                         delta_t=1.0/4096,
                         f_lower=25)


#hp.taper()


fig = plt.figure()
plt.plot(hp.sample_times, hp, label=apx)
plt.ylabel('Strain')
plt.xlabel('Time (s)')
plt.legend()
st.pyplot(fig)



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




