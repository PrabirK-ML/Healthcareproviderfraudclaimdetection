from flask import Flask
from flask import request,jsonify,make_response
import pickle
import pandas as pd
import numpy as np
import sklearn
from flasgger import Swagger
import copy

app=Flask(__name__)
Swagger(app)
@app.route('/')
def hello():
    return "Hello From Flasgger"

def impute_zero(column):
  if pd.isnull(column[0]):
    return 0.0
  return column[0]


def transform_data(transf,indata):
  ''' This transform function maps input data with values from transform data '''
  #data=indata.copy(deep=True)
  data=copy.deepcopy(indata)

  data[data.columns[0]]=data[data.columns[0]].map(transf.set_index(transf.columns[0])[transf.columns[-1]])
  data[data.columns[0]]=data[[data.columns[0]]].apply(impute_zero,axis=1)
  #sample=pd.DataFrame(df1['Name'].map(df2.set_index('Name')['Marks']))
  #sample[sample.columns[0]]=sample[[sample.columns[0]]].apply(impute_zero,axis=1)
  sample=data[data.columns[0]]
  return sample

@app.route('/validate')
def validate():
    """ Health Care Provider Fraud Claim Detection
    ---
    parameters:
        - name: Admission_Duration
          in: query
          type: string
          required: true
        - name: Attending_physician_code
          in: query
          type: string
          required: true
        - name: BeneID
          in: query
          type: string
          required: true
        - name: ChronicCond_KidneyDisease
          in: query
          type: string
          required: true
        - name: ClmAdmitDiagnosisCode
          in: query
          type: string
          required: true
        - name: ClaimDiagnosticCode1_clmdc1
          in: query
          type: string
          required: true
        - name: clmdc2
          in: query
          type: string
          required: true
        - name: clmdc3
          in: query
          type: string
          required: true
        - name: clmdc4
          in: query
          type: string
          required: true
        - name: clmdc5
          in: query
          type: string
          required: true
        - name: clmdc6
          in: query
          type: string
          required: true
        - name: clmdc7
          in: query
          type: string
          required: true
        - name: country
          in: query
          type: string
          required: true
        - name: DeductibleAmtPaid
          in: query
          type: string
          required: true
        - name: DiagnosisGroupCode
          in: query
          type: string
          required: true
        - name: operating_physician
          in: query
          type: string
          required: true
        - name: other_physician
          in: query
          type: string
          required: true
        - name: provider
          in: query
          type: string
          required: true
        - name: RenalDiseaseIndicator
          in: query
          type: string
          required: true
        - name: state
          in: query
          type: string
          required: true
    responses:
        200:
            description: Result
    """
      
    adm_dur=request.args.get('Admission_Duration')
    ap=request.args.get('Attending_physician_code')
    beneid=request.args.get('BeneID')
    ChronicCond_KidneyDisease=request.args.get('ChronicCond_KidneyDisease')
    ClmAdmitDiagnosisCode=request.args.get('ClmAdmitDiagnosisCode')
    clmdc1=request.args.get('ClaimDiagnosticCode1_clmdc1')
    clmdc2=request.args.get('clmdc2')
    clmdc3=request.args.get('clmdc3')
    clmdc4=request.args.get('clmdc4')
    clmdc5=request.args.get('clmdc5')
    clmdc6=request.args.get('clmdc6')
    clmdc7=request.args.get('clmdc7')
    country=request.args.get('country')
    DeductibleAmtPaid=request.args.get('DeductibleAmtPaid')
    DiagnosisGroupCode=request.args.get('DiagnosisGroupCode')
    op=request.args.get('operating_physician')
    ot=request.args.get('other_physician')
    provider=request.args.get('provider')
    RenalDiseaseIndicator=request.args.get('RenalDiseaseIndicator')
    state=request.args.get('state')


    x_input=pd.DataFrame()
    x_input['col1']=transform_data(beneid_fit,pd.DataFrame([beneid]))
    x_input['col2']=transform_data(provider_fit,pd.DataFrame([provider]))
    x_input['col3']=transform_data(ap_fit,pd.DataFrame([ap]))
    x_input['col4']=transform_data(op_fit,pd.DataFrame([op]))
    x_input['col5']=transform_data(ot_fit,pd.DataFrame([ot]))
    x_input['col6']=transform_data(state_fit,pd.DataFrame([state]))
    x_input['col7']=transform_data(country_fit,pd.DataFrame([country]))
    x_input['col8']=transform_data(DiagnosisGroupCode_fit,pd.DataFrame([DiagnosisGroupCode]))
    x_input['col9']=transform_data(ClmAdmitDiagnosisCode_fit,pd.DataFrame([ClmAdmitDiagnosisCode]))
    x_input['col10']=transform_data(clmdc1_fit,pd.DataFrame([clmdc1]))
    x_input['col11']=transform_data(clmdc2_fit,pd.DataFrame([clmdc2]))
    x_input['col12']=transform_data(clmdc3_fit,pd.DataFrame([clmdc3]))
    x_input['col13']=transform_data(clmdc4_fit,pd.DataFrame([clmdc4]))
    x_input['col14']=transform_data(clmdc5_fit,pd.DataFrame([clmdc5]))
    x_input['col15']=transform_data(clmdc6_fit,pd.DataFrame([clmdc6]))
    x_input['col16']=transform_data(clmdc7_fit,pd.DataFrame([clmdc7]))
    x_rdi=pd.DataFrame([[RenalDiseaseIndicator,ChronicCond_KidneyDisease,DeductibleAmtPaid,adm_dur]],columns=['RenalDiseaseIndicator','KidneyDisease','DeductibleAmtPaid','adm_dur'])
    x_rdi['RenalDiseaseIndicator']=RenalDiseaseIndicator_enc_fit.transform(x_rdi[['RenalDiseaseIndicator']])
    x_rdi['KidneyDisease']=ChronicCond_KidneyDisease_enc_fit.transform(x_rdi[['KidneyDisease']])
    x_rdi['DeductibleAmtPaid']=DeductibleAmtPaid_norm_fit.transform(x_rdi[['DeductibleAmtPaid']])
    x_rdi['adm_dur']=adm_dur_norm_fit.transform(x_rdi[['adm_dur']])
    print(x_rdi)
    x_input['col17']=x_rdi['RenalDiseaseIndicator']
    x_input['col18']=x_rdi['KidneyDisease']
    print(pd.DataFrame(DeductibleAmtPaid_norm_fit.transform(pd.DataFrame([DeductibleAmtPaid]))))
    x_input['col19']=x_rdi['DeductibleAmtPaid']
    x_input['col20']=x_rdi['adm_dur']

    predict_result=classifier.predict(x_input)
    if predict_result==0:
        predict_result='Genuine'
    else:
        predict_result='Fraud'

    response=make_response(jsonify({'Finding':predict_result}))
    
    
    return response



if __name__=="__main__":
    classifier=open('./required_files/model.pkl','rb')
    classifier=pickle.load(classifier)
    #print(classifier)
    adm_dur_norm_fit=open('./required_files/Adm_duration_norm_fit.pkl','rb')
    adm_dur_norm_fit=pickle.load(adm_dur_norm_fit)
    
    ap_fit=open('./required_files/ap_fit.pkl','rb')
    ap_fit=pickle.load(ap_fit)
    
    beneid_fit=open('./required_files/beneid_fit.pkl','rb')
    beneid_fit=pickle.load(beneid_fit)
    
    ChronicCond_KidneyDisease_enc_fit=open('./required_files/ChronicCond_KidneyDisease_enc_fit.pkl','rb')
    ChronicCond_KidneyDisease_enc_fit=pickle.load(ChronicCond_KidneyDisease_enc_fit)
    
    ClmAdmitDiagnosisCode_fit=open('./required_files/ClmAdmitDiagnosisCode_fit.pkl','rb')
    ClmAdmitDiagnosisCode_fit=pickle.load(ClmAdmitDiagnosisCode_fit)
    
    clmdc1_fit=open('./required_files/clmdc1_fit.pkl','rb')
    clmdc1_fit=pickle.load(clmdc1_fit)
    
    clmdc2_fit=open('./required_files/clmdc1_fit.pkl','rb')
    clmdc2_fit=pickle.load(clmdc2_fit)
    
    clmdc3_fit=open('./required_files/clmdc1_fit.pkl','rb')
    clmdc3_fit=pickle.load(clmdc3_fit)
    
    clmdc4_fit=open('./required_files/clmdc1_fit.pkl','rb')
    clmdc4_fit=pickle.load(clmdc4_fit)
    
    clmdc5_fit=open('./required_files/clmdc1_fit.pkl','rb')
    clmdc5_fit=pickle.load(clmdc5_fit)
    
    clmdc6_fit=open('./required_files/clmdc1_fit.pkl','rb')
    clmdc6_fit=pickle.load(clmdc6_fit)
    
    clmdc7_fit=open('./required_files/clmdc1_fit.pkl','rb')
    clmdc7_fit=pickle.load(clmdc7_fit)
    
    country_fit=open('./required_files/country_fit.pkl','rb')
    country_fit=pickle.load(country_fit)
    
    DeductibleAmtPaid_norm_fit=open('./required_files/DeductibleAmtPaid_norm_fit.pkl','rb')
    DeductibleAmtPaid_norm_fit=pickle.load(DeductibleAmtPaid_norm_fit)
    
    DiagnosisGroupCode_fit=open('./required_files/DiagnosisGroupCode_fit.pkl','rb')
    DiagnosisGroupCode_fit=pickle.load(DiagnosisGroupCode_fit)
    
    op_fit=open('./required_files/op_fit.pkl','rb')
    op_fit=pickle.load(op_fit)
    
    ot_fit=open('./required_files/ot_fit.pkl','rb')
    ot_fit=pickle.load(ot_fit)
    
    provider_fit=open('./required_files/provider_fit.pkl','rb')
    provider_fit=pickle.load(provider_fit)
    
    RenalDiseaseIndicator_enc_fit=open('./required_files/RenalDiseaseIndicator_enc_fit.pkl','rb')
    RenalDiseaseIndicator_enc_fit=pickle.load(RenalDiseaseIndicator_enc_fit)
    
    state_fit=open('./required_files/state_fit.pkl','rb')
    state_fit=pickle.load(state_fit)

    app.run(host='0.0.0.0')
