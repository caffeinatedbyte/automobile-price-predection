import streamlit as st
import warnings
import numpy as np
warnings.filterwarnings('ignore')
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
import pickle
st.title("Automobile data case study")
#To accept data from user 
#numeric data
symbolizing=[-2,-1,0,1,2,3]
sym=st.selectbox('Symbolic',symbolizing,index=symbolizing.index(-2))
nl=st.number_input('Noarmalised losses',value=0,step=1)
w=st.number_input('Width (mm)',format='%d',value=0,step=1)   
h=st.number_input('Height (mm)',format='%d',value=0,step=1)
es=st.number_input('Engine Size (cc)',format='%d',value=0,step=1)#integer
hp=st.number_input('Horse power (bhp)',format='%d',value=0,step=1)#integer
cm=st.number_input('City Mileage (km/l)',format='%d',value=0,step=1)
hm=st.number_input('Highway Mileage (km/l)',format='%d',value=0,step=1)

#string Data
brand=['alfa-romero', 'audi', 'bmw', 'chevrolet', 'dodge', 'honda',
       'isuzu', 'jaguar', 'mazda', 'mercedes-benz', 'mercury',
       'mitsubishi', 'nissan', 'peugot', 'plymouth', 'porsche', 'renault',
       'saab', 'subaru', 'toyota', 'volkswagen', 'volvo']
make=st.selectbox('Select car brand',brand,index=brand.index('alfa-romero'))#object type
fuel=['petrol', 'diesel']
ft=st.selectbox('Select fuel type',fuel,index=fuel.index('petrol'))#object type
body=['convertible', 'hatchback', 'sedan', 'wagon', 'hardtop']
bs=st.selectbox('Body Style',body,index=body.index('convertible'))#object type
drive=['rwd', 'fwd', '4wd']
dw=st.selectbox('Drive wheels',drive,index=drive.index('rwd'))#object type
enginel=['front', 'rear']
el=st.selectbox('Engine Location',enginel,index=enginel.index('front'))#object type
enginet=['dohc', 'ohcv', 'ohc', 'l', 'rotor', 'ohcf', 'dohcv']
et=st.selectbox('Engine type',enginet,index=enginet.index('dohc'))#object type


if st.button("Predict"):
    try:
        # Load models and encoders
        with open('scale1.pkl', 'rb') as file1, open('model1.pkl', 'rb') as file2, \
                open("LabelEncoder1.pkl", "rb") as file3_1, open("LabelEncoder2.pkl", "rb") as file3_2, \
                open("LabelEncoder3.pkl", "rb") as file3_3, open("LabelEncoder4.pkl", "rb") as file3_4, \
                open("LabelEncoder5.pkl", "rb") as file3_5, open("LabelEncoder6.pkl", "rb") as file3_6:
            # Read data from files
            scale = pickle.load(file1)
            model = pickle.load(file2)
            e_label1 = pickle.load(file3_1)
            e_label2 = pickle.load(file3_2)
            e_label3 = pickle.load(file3_3)
            e_label4 = pickle.load(file3_4)
            e_label5 = pickle.load(file3_5)
            e_label6 = pickle.load(file3_6)

            # Transform input data
            x1 = e_label1.transform([make])[0]
            x2 = e_label2.transform([ft])[0]
            x3 = e_label3.transform([bs])[0]
            x4 = e_label4.transform([dw])[0]
            x5 = e_label5.transform([el])[0]
            x6 = e_label6.transform([et])[0]

            # Convert symbolic value to float
            sym = float(sym)

            # Create feature vector
            xf = [x1, x2, x3, x4, x5, x6, sym, nl, w, h, es, hp, cm, hm]

            # Verify data types and shapes
            for item in xf:
                print(type(item), item)

            # Transform features
            X = np.array([xf]).astype(float)

            # Verify shape
            print(X.shape)

            # Predict
            Y_pred = model.predict(X)
            st.write("Predicted price = ", Y_pred)

    except FileNotFoundError:
        st.error("Error: One or more required files not found.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
