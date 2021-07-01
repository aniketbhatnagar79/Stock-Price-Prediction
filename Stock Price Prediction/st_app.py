import streamlit as st

import pandas as pd
import plotly.express as px


import numpy as np
from tensorflow import keras
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM




st.title('Stock Price Prediction')

csv_file = st.file_uploader('Upload dataset', 'csv')

if csv_file is not None:
    st.subheader('Dataset')
    df = pd.read_csv(csv_file)
    df=df.dropna()
    st.dataframe(df)


    columns = df.columns.tolist()
    
    st.subheader('Data Plot')
    columns.append('Choose')
    xc = st.selectbox('Select a Column for X-axis', columns, index=len(columns) - 1)

    cols = st.multiselect('Select Columns for Y-axis', columns)

    if xc != columns[-1]:
        if len(cols) >= 1 and xc not in cols:
            fig = px.line(df, x=xc, y=cols)
            st.plotly_chart(fig, use_container_width=True)
        
        prediction_target=st.selectbox('Select Prediction Target', columns,index=len(columns)-1)

        
        if prediction_target!=columns[-1]:
            
            data={xc:df[xc],prediction_target:df[prediction_target]}

            st.dataframe(pd.DataFrame(data))

            k=df[prediction_target][-10:].values 

            

            # Extracting the closing prices of each day
            FullData=df[[prediction_target]].values
            

            # Feature Scaling for fast training of neural networks
            from sklearn.preprocessing import StandardScaler, MinMaxScaler

            # Choosing between Standardization or normalization
            #sc = StandardScaler()
            sc=MinMaxScaler()

            DataScaler = sc.fit(FullData)
            X=DataScaler.transform(FullData)

            # split into samples
            X_samples = list()
            y_samples = list()

            NumerOfRows = len(X)
            TimeSteps=10  # next day's Price Prediction is based on last how many past day's prices

            # Iterate thru the values to create combinations
            for i in range(TimeSteps , NumerOfRows , 1):
                x_sample = X[i-TimeSteps:i]
                y_sample = X[i]
                X_samples.append(x_sample)
                y_samples.append(y_sample)

            ################################################
            # Reshape the Input as a 3D (number of samples, Time Steps, Features)
            X_data=np.array(X_samples)
            X_data=X_data.reshape(X_data.shape[0],X_data.shape[1], 1)
            

            # We do not reshape y as a 3D data  as it is supposed to be a single column only
            y_data=np.array(y_samples)
            y_data=y_data.reshape(y_data.shape[0], 1)
            
            

            TestingRecords=5

            # Splitting the data into train and test
            X_train=X_data[:-TestingRecords]
            X_test=X_data[-TestingRecords:]
            y_train=y_data[:-TestingRecords]
            y_test=y_data[-TestingRecords:]

            TimeSteps=X_train.shape[1]
            TotalFeatures=X_train.shape[2]

            # Initialising the RNN
            regressor = Sequential()

            # Adding the First input hidden layer and the LSTM layer
            # return_sequences = True, means the output of every time step to be shared with hidden next layer
            regressor.add(LSTM(units = 10, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))

            # Adding the Second Second hidden layer and the LSTM layer
            regressor.add(LSTM(units = 5, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))

            # Adding the Second Third hidden layer and the LSTM layer
            regressor.add(LSTM(units = 5, activation = 'relu', return_sequences=False ))


            # Adding the output layer
            regressor.add(Dense(units = 1))

            # Compiling the RNN
            regressor.compile( optimizer = 'adam', loss = 'mean_squared_error')

            ##################################################

            import time
            from tensorflow import keras
            # Measuring the time taken by the model to train
            StartTime=time.time()

            # Fitting the RNN to the Training set
            #regressor.fit(X_train, y_train, batch_size = 5, epochs = 100)

            
            with st.spinner('Predecting Stock Price...'):
                
                
                history =regressor.fit(X_train, y_train, batch_size = 5, epochs = 100)

                # Generating predictions on full data
                TrainPredictions=DataScaler.inverse_transform(regressor.predict(X_train))
                TestPredictions=DataScaler.inverse_transform(regressor.predict(X_test))

                FullDataPredictions=np.append(TrainPredictions, TestPredictions)
                FullDataOrig=FullData[TimeSteps:]
  
                # Last 10 days prices
                
                Last10Days=k
                # Normalizing the data just like we did for training the model
                Last10Days=DataScaler.transform(Last10Days.reshape(-1,1))

                # Changing the shape of the data to 3D
                # Choosing TimeSteps as 10 because we have used the same for training
                NumSamples=1
                TimeSteps=10
                NumFeatures=1
                Last10Days=Last10Days.reshape(NumSamples,TimeSteps,NumFeatures)

                #############################

                # Making predictions on data
                predicted_Price = regressor.predict(Last10Days)
                predicted_Price = DataScaler.inverse_transform(predicted_Price)

                predicted_Price_string='Price For Tomorrow Will be: '
                    
                
                loss = history.history['loss'][-1]
               
            if loss > 0.001:
                st.write('Loss Exceeded than Acceptable , Please Predict Again')

            else :

                st.write(predicted_Price_string)
                st.success(str(predicted_Price[0][0])+' Rs')
                
            



