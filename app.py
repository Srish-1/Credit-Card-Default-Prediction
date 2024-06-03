# Importing all required modules 
import bz2 as bz2
import datetime as dt
import pickle
import webbrowser as wb

import numpy as np
import pandas as pd
import streamlit as sl
import streamlit.components.v1 as com
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# Function to unpickle 
def decompress_pickle(file):
  data = bz2.BZ2File(file, 'rb')
  data = pickle.load(data)
  return data

# Function use to open link which will download the demo excel file
def open_link(str):
  wb.open(str)

# Unpacking Scaler pkl file
model = decompress_pickle('ccdp.pbz2')
S_file = open('scaler.pkl','rb')
scaler = pickle.load(S_file)

# creating 4 Dicts which is used to convert to str to int befor giveing input to ML module
S_DICT = {'Male':1,'Female':2}
M_DICT = {'Married':1,'Single':2,'Others':3}
E_DICT ={'Graduate school':1,'University':2,'High school':3,'Others':4}
PAY_DICT= {'Zero Bill':0,'Paid duly':-1,'1 Month Delay':1,'2 Months Delay':2,'3 Months Delay':3,'4 Months Delay':4,'5 Months Delay':5,'6 Months Delay':6,'7 Months Delay':7,'8 Months Delay':8,'9 Months & Above Delay':9}

# while loop for Dynamic Months
n=0
Dynamic_months={}
current_month=dt.datetime.now()
while n < 6:
    month=current_month.strftime('%B')
    Dynamic_months['m{0}'.format(n)]=month
    current_month=current_month-dt.timedelta(days=27)
    n=n+1

# Function which handels multi transactions
def multi_cust(file):
  if file:
    # Reading as Pandas dataframe
    df = pd.read_excel(file)
    # Preprocessing
    cust_ID_DF=pd.DataFrame()
    df.drop(0,axis=0,inplace=True)
    cust_ID_DF['customer_id/Name'] = df['Customer_ID/Name']
    df.drop('Customer_ID/Name',axis=1,inplace=True)
    df['AVG_BILL_AMT'] = ((df['1st Month.1']+df['2nd Month.1']+df['3rd Month.1']+df['4th Month.1']+df['5th Month.1']+df['6th Month.1'])/6)
    df['SEX'] = df['SEX'].apply(lambda x: S_DICT[x])
    df['EDUCATION'] = df['EDUCATION'].apply(lambda x: E_DICT[x])
    df['MARRIAGE'] = df['MARRIAGE'].apply(lambda x: M_DICT[x])
    df['1st Month'] = df['1st Month'].apply(lambda x: PAY_DICT[x])
    df['2nd Month'] = df['2nd Month'].apply(lambda x: PAY_DICT[x])
    df['3rd Month'] = df['3rd Month'].apply(lambda x: PAY_DICT[x])
    df['4th Month'] = df['4th Month'].apply(lambda x: PAY_DICT[x])
    df['5th Month'] = df['5th Month'].apply(lambda x: PAY_DICT[x])
    df['6th Month'] = df['6th Month'].apply(lambda x: PAY_DICT[x])
    # Scaling the model
    df_scaled = scaler.transform(df)
    # Pridecting
    multi_pred = model.predict(df_scaled)
    cust_ID_DF['default']=multi_pred
    cust_ID_DF['Status for next month']=cust_ID_DF['default'].apply(lambda x : 'Repay' if x == 0 else 'Default')
    cust_ID_DF.drop('default',axis=1,inplace=True)
    # Saving excel with only Customer name/ID with prediction
    data_frame= cust_ID_DF.to_csv()
    # Showing on the platform
    sl.table(cust_ID_DF)
    # Download button for the file
    sl.download_button(label='Download tabel',data=data_frame,mime='text/csv',file_name='Bill Payment Status for Next month')

 # Function to print out put which also converts numeric output from ML module to understandable STR 
def pred_out(num):
    if num == 1:
      sl.warning('This customer IS LIKELY TO DEFAULT next month..!')
    else:
      sl.success('This customer IS NOT LIKELY TO DEFAULT next month..!')

# Titel 
sl.title('Creadit Card Default Predction')

# Selection 
rad_b=sl.radio('Please select that you want give Single or Multiple customers data',options=['Singel','Multipel'])

# Body of the page using FORM
def main():
  if rad_b == 'Singel':
    form = sl.form('Customer Details')
    C_name = form.text_input('Name')
    col1,col2,col3 = form.columns(3)
    C_sex= S_DICT[col1.radio('Sex',options=['Male','Female'])]
    C_marg = M_DICT[col2.radio('Marital status',options=['Married','Single','Others'])]
    C_edu = E_DICT[col3.radio('Education',options=['Graduate school','University','High school','Others'])]
    col1,col2 = form.columns(2)
    C_age = col1.number_input('Age')
    C_Limit = col2.number_input('Limit Amout in $')
    # PAY input
    
    L_opt = ['Zero Bill','Paid duly','1 Month Delay','2 Months Delay','3 Months Delay','4 Months Delay','5 Months Delay','6 Months Delay','7 Months Delay','8 Months Delay','9 Months & Above Delay']
    form.subheader('Repayment Status for last 6 months')
    col1,col2,col3=form.columns(3)
    pay1=PAY_DICT[col1.selectbox(label=Dynamic_months['m5'],options=L_opt,key=1)]
    pay2=PAY_DICT[col2.selectbox(label=Dynamic_months['m4'],options=L_opt,key=2)]
    pay3=PAY_DICT[col3.selectbox(label=Dynamic_months['m3'],options=L_opt,key=3)]
    col4,col5,col6=form.columns(3)
    pay4=PAY_DICT[col4.selectbox(label=Dynamic_months['m2'],options=L_opt,key=4)]
    pay5=PAY_DICT[col5.selectbox(label=Dynamic_months['m1'],options=L_opt,key=5)]
    pay6=PAY_DICT[col6.selectbox(label=Dynamic_months['m0'],options=L_opt,key=6)]
    
    form.subheader('Bill Amount in the respective months in $')
    col14,col15,col16=form.columns(3)
    BAmt1=col14.number_input(label=(Dynamic_months['m5']+' (in $)'),key=7)
    BAmt2=col15.number_input(label=(Dynamic_months['m4']+' (in $)'),key=8)
    BAmt3=col16.number_input(label=(Dynamic_months['m3']+' (in $)'),key=9)
    col17,col18,col19=form.columns(3)
    BAmt4=col17.number_input(label=(Dynamic_months['m2']+' (in $)'),key=10)
    BAmt5=col18.number_input(label=(Dynamic_months['m1']+' (in $)'),key=11)
    BAmt6=col19.number_input(label=(Dynamic_months['m0']+' (in $)'),key=12)
    
    form.subheader('Amount paid for previous bill in $')
    col7,col8,col9=form.columns(3)
    PAmt1=col7.number_input(label=(Dynamic_months['m5']+' (in $)'),key=13)
    PAmt2=col8.number_input(label=(Dynamic_months['m4']+' (in $)'),key=14)
    PAmt3=col9.number_input(label=(Dynamic_months['m3']+' (in $)'),key=15)
    col11,col12,col13=form.columns(3)
    PAmt4=col11.number_input(label=(Dynamic_months['m2']+' (in $)'),key=16)
    PAmt5=col12.number_input(label=(Dynamic_months['m1']+' (in $)'),key=17)
    PAmt6=col13.number_input(label=(Dynamic_months['m0']+' (in $)'),key=18)
    
    # Creating new feature Average Bill Amount 
    avg_bill_amt = np.mean(BAmt1+BAmt2+BAmt3+BAmt4+BAmt5+BAmt6)
    features=[C_Limit,C_sex,C_edu,C_marg,C_age,pay1,pay2,pay3,pay4,pay5,pay6,BAmt1,BAmt2,BAmt3,BAmt4,BAmt5,BAmt6,PAmt1,PAmt2,PAmt3,PAmt4,PAmt5,PAmt6,avg_bill_amt]
    features_s = scaler.transform(np.array(features,ndmin=2))
    pred = model.predict(features_s)

    P_satus=form.form_submit_button("Predict")
    # If predict button clicked it will predict
    if P_satus:
      pred_out(pred)

  else:
    # Multi transaction 
    sl.subheader('Please Download the Demo excel file')
    sl.text('Note:- enter the details of custome, save & Upload, Dont change to format.!')
    # HTML code for downloading demo file 
    com.html(f"""<button onclick="window.location.href='https://drive.google.com/uc?export=download&id=10aYBUF50jjAWvi-ukZZE2Q6_8pbLoUon';">
                      Download Demo File</button>""",height=30)
    multi_cust(sl.file_uploader('Please Upload Excel file'))


if __name__ == '__main__':
  main()
  
    