import streamlit as st
import pickle
import numpy as np

pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))
st.title("laptop Predictor")
company = st.selectbox('Brand', df['Company'].unique())
Typename = st.selectbox('Type', df['TypeName'].unique())
# screen size
screen_size = st.number_input('Screen Size')
# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])



# resolution
X_res = st.selectbox('X_res',df['X_res'].unique())
Y_res = st.selectbox('Y_res',df['Y_res'].unique())

#cpu
cpu = st.selectbox('CPU',df['Cpu brand'].unique())

gpu = st.selectbox('GPU',df['Processor'].unique())

os = st.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0
    query = np.array([company,Typename,screen_size,ram,weight,touchscreen,ips,X_res,Y_res,cpu,gpu,os])

    query = query.reshape(1,12)
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))