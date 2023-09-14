#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.inspection import permutation_importance

data = pd.read_csv("dogfinal3.csv")

X = data.drop(["diseases"], axis=1)
y = data["diseases"]

#LIST OF SYMPTOMS OF EACH DISEASE

class_rabies = data[data['diseases'] == 'rabies']
rabies_sympts = class_rabies.columns[(class_rabies == 1).any()]

class_caninedistemper = data[data['diseases'] == 'caninedistemper']
caninedistemper_sympts = class_caninedistemper.columns[(class_caninedistemper == 1).any()]

class_leptospirosis = data[data['diseases'] == 'leptospirosis']
leptospirosis_sympts = class_leptospirosis.columns[(class_leptospirosis == 1).any()]

class_kennelcough = data[data['diseases'] == 'kennelcough']
kennelcough_sympts = class_kennelcough.columns[(class_kennelcough == 1).any()]

class_kidneydisease = data[data['diseases'] == 'kidneydisease']
kidneydisease_sympts = class_kidneydisease.columns[(class_kidneydisease == 1).any()]

class_heartworm = data[data['diseases'] == 'heartworm']
heartworm_sympts = class_heartworm.columns[(class_heartworm == 1).any()]

class_canineparvovirus = data[data['diseases'] == 'canineparvovirus']
canineparvovirus_sympts = class_canineparvovirus.columns[(class_canineparvovirus == 1).any()]

df =pd.DataFrame(data=None, columns=('symptoms', *data['diseases'].unique()))
df['symptoms']=data.columns.drop('diseases')

df.rename(columns = {'caninedistemper':'Canine Distemper', 'canineparvovirus':'Canine Parvovirus',
                     'heartworm':'Heartworm', 'kennelcough':'Kennel Cough','rabies':'Rabies',
                     'leptospirosis':'Leptospirosis', 'kidneydisease':'Kidney Disease'}, inplace = True)

#sev values
caninedistemper= {'Excellent': ['difficultyinbreathing','seizures', 'depression'],
                 'Good': ['fever', 'vomiting', 'paralysis', 'reducedappetite', 'coughing', 'dischargefromeyes',
                             'nasaldischarge', 'lethargy', 'sneezing', 'diarrhea', 'pain', 'skinsores',
                             'inflammation_eyes', 'anorexia'], 
                 'Fair': ['hyperkeratosis']}
rabies={'Excellent':['hydrophobia', 'seizures', 'paralysis', 'difficultyinswallowing', 
                            'foamingatmouth', 'aggression', 'highlyexcitable'],
                 'Good': ['fever', 'vomiting', 'reduced appetite', 'lethargy', 'lameness', 'irritable'],
                 'Fair': ['pica','exesssalivation']}
leptospirosis={'Excellent': ['difficultyinbreathing', 'jaundice', 'bloodinurine'],
                 'Good': ['fever', 'vomiting','reducedappetite', 'coughing', 'nasaldischarge',
                             'lethargy', 'diarrhea', 'depression', 'weakness', 'stiffness', 'limping', 'dehydration'], 
                 'Fair': ['increasedthirst', 'increasedurination', 'shivering', 'laziness', 'bloodinstool']}
kennelcough={'Good': ['coughing', 'difficultyinbreathing', 'gagging'], 
                 'Fair': ['fever', 'vomiting', 'reducedappetite', 'dischargefromeyes',
                              'nasaldischarge', 'lethargy', 'sneezing', 'weakness', 'reversesneezing']}
kidneydisease ={'Excellent': ['seizures', 'bloodinurine', 'depression'],
                 'Good': ['vomiting', 'reducedappetite', 'lethargy', 'diarrhea', 'weightloss', 'weakness', 'lameness', 'increasedthirst',
                             'increasedurination', 'palegums', 'ulcersinmouth', 'badbreath'], 
                 'Fair': [ 'decreasedthirst','decreasedurination']}
heartworm= {'Excellent': ['difficultyinbreathing', 'fainting', 'rapidheartbeat'],
                 'Good': ['coughing', 'reducedappetite', 'lethargy', 'weightloss', 'fatigue',
                             'swollenbelly', 'laziness'], 
                 'Fair': ['anemia']}
canineparvovirus= {'Excellent': ['rapidheartbeat', 'bloodystool', 'diarrhea', 'vomiting', 'dehydration'],
                 'Good': ['fever', 'reducedappetite', 'lethargy', 'depression', 'pain', 'inflammation_eyes', 
                             'anorexia', 'weightloss','weakness', 'inflammation_mouth'], 
                 'Fair': ['vomiting', ' diarrhea']}

#storing values

values = {'Excellent': 3, 'Good': 2, 'Fair': 1}
for i, d in enumerate([caninedistemper]):
    for key in d.keys():
        for symptom in d[key]:
            df.loc[df['symptoms'] == symptom, 'Canine Distemper'] = values[key]
for i, d in enumerate([rabies]):
    for key in d.keys():
        for symptom in d[key]:
            df.loc[df['symptoms'] == symptom, 'Rabies'] = values[key]
for i, d in enumerate([leptospirosis]):
    for key in d.keys():
        for symptom in d[key]:
            df.loc[df['symptoms'] == symptom, 'Leptospirosis'] = values[key]
for i, d in enumerate([kidneydisease]):
    for key in d.keys():
        for symptom in d[key]:
            df.loc[df['symptoms'] == symptom, 'Kidney Disease'] = values[key]
for i, d in enumerate([heartworm]):
    for key in d.keys():
        for symptom in d[key]:
            df.loc[df['symptoms'] == symptom, 'Heartworm'] = values[key]
for i, d in enumerate([kennelcough]):
    for key in d.keys():
        for symptom in d[key]:
            df.loc[df['symptoms'] == symptom, 'Kennel Cough'] = values[key]  
for i, d in enumerate([canineparvovirus]):
    for key in d.keys():
        for symptom in d[key]:
            df.loc[df['symptoms'] == symptom, 'Canine Parvovirus'] = values[key]
df.fillna(0, inplace=True)    

###################
df = df.set_index('symptoms')

#my def
def classify_disease(symptoms, disease):
    count_3 = 0
    count_2 = 0
    count_1 = 0
    weight_3 = 0.5
    weight_2 = 0.3
    weight_1 = 0.2
    over = 0

    # loop through the list of symptoms
    for symptom in symptoms:
        value = df.loc[symptom, disease]
        # check if the value is not 0
        if value != 0:
            over += 1
            # check if any symptom has a value of 3
            if value == 3:
                count_3 += 1
            elif value == 2:
                count_2 += 1
            elif value == 1:
                count_1 += 1
    total_3 = (df[disease] > 0).sum()

    if over != 0 and total_3 != 0:
        final_percentage = round((count_3/over*weight_3 + count_2/over*weight_2 + count_1/over*weight_1) * 100 * (count_3/total_3) , 2)
    else:
        final_percentage = 0

    if count_3 > 0:
        return f"Poor"
    if final_percentage >= 80:
        return f"Poor"
    if final_percentage >= 50:
        return f"Fair"
    if final_percentage >= 20:
        return f"Good"
    if final_percentage <= 10:
        return f"Excellent"

    
#flask

from flask import Flask, jsonify, request, session
from flask import redirect, url_for
import pickle

app = Flask(__name__)
with open("model.pkl","rb") as f:
    model = pickle.load(f)

from flask import render_template
#import warnings 
#warnings.filterwarnings('ignore')




@app.route("/")
def home():
    return render_template("home2.html")

@app.route("/get_names", methods=["post"])
def get_names():
    your_name = request.form.get("your_name")
    pet_name = request.form.get("pet_name")
    
    return render_template("home2.html", your_name=your_name, pet_name=pet_name)

@app.route("/team", methods=["GET"])
def team():
    return render_template("team.html")

@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html")

                           
@app.route("/")
def form():
    return render_template("form.html")


#predict


import numpy as np
import pandas as pd

@app.route("/predict", methods=["GET", "POST"])
def predict():
    
    fever = request.form.get("fever")
    vomiting = request.form.get("vomiting")
    paralysis = request.form.get("paralysis")
    reducedappetite = request.form.get("reducedappetite")
    coughing = request.form.get("coughing")
    dischargefromeyes = request.form.get("dischargefromeyes")
    hyperkeratosis = request.form.get("hyperkeratosis")
    nasaldischarge = request.form.get("nasaldischarge")
    lethargy = request.form.get("lethargy")
    sneezing = request.form.get("sneezing")
    diarrhea = request.form.get("diarrhea")
    depression = request.form.get("depression")
    difficultyinbreathing = request.form.get("difficultyinbreathing")
    pain = request.form.get("pain")
    skinsores = request.form.get("skinsores")
    inflammation_eyes = request.form.get("inflammation-eyes")
    anorexia = request.form.get("anorexia")
    seizures = request.form.get("seizures")
    dehydration = request.form.get("dehydration")
    weightloss = request.form.get("weightloss")
    bloodystool = request.form.get("bloodystool")
    weakness = request.form.get("weakness")
    inflammation_mouth = request.form.get("inflammation-mouth")
    rapidheartbeat = request.form.get("rapidheartbeat")
    fatigue = request.form.get("fatigue")
    swollenbelly = request.form.get("swollenbelly")
    laziness = request.form.get("laziness")
    anemia = request.form.get("palegums")
    fainting = request.form.get("fainting")
    reversesneezing = request.form.get("reversesneezing")
    gagging = request.form.get("gagging")
    lameness = request.form.get("lameness")
    stiffness = request.form.get("stiffness")
    limping = request.form.get("limping")
    increasedthirst = request.form.get("increasedthirst")
    increasedurination = request.form.get("increasedurination")
    excesssalivation = request.form.get("excesssalivation")
    aggression = request.form.get("aggression")
    foamingatmouth = request.form.get("foamingatmouth")
    difficultyinswallowing = request.form.get("difficultyinswallowing")
    irritable = request.form.get("irritable")
    pica = request.form.get("pica")
    hydrophobia = request.form.get("hydrophobia")
    highlyexcitable = request.form.get("highlyexcitable")
    shivering = request.form.get("shivering")
    jaundice = request.form.get("jaundice")
    decreasedthirst = request.form.get("decreasedthirst")
    decreasedurination = request.form.get("decreasedurination")
    bloodinurine = request.form.get("bloodinurine")
    palegums = request.form.get("palegums")
    ulcersinmouth = request.form.get("ulcersinmouth")
    badbreath = request.form.get("badbreath")
    feature= [fever, vomiting, paralysis, reducedappetite,coughing, dischargefromeyes,
                         hyperkeratosis, nasaldischarge,lethargy, 
                         sneezing, diarrhea, depression,difficultyinbreathing, pain, skinsores, 
                         inflammation_eyes,anorexia, seizures, dehydration, weightloss, bloodystool,
                         weakness, inflammation_mouth, rapidheartbeat, fatigue,swollenbelly, laziness, anemia, 
                         fainting, reversesneezing, gagging,lameness, stiffness, 
                         limping, increasedthirst,increasedurination, excesssalivation, aggression,
                         foamingatmouth, difficultyinswallowing, irritable, pica,hydrophobia, 
                         highlyexcitable,shivering, jaundice, 
                         decreasedthirst, decreasedurination, bloodinurine,palegums, ulcersinmouth, badbreath]
    feature = np.array(feature, dtype=float)
    # Replace NaN values with zeroes
    
    for i in range(feature.shape[0]):
        if np.isnan(feature[i]):
            feature[i]= 0
    feature1 = pd.DataFrame(data=None, columns=data.columns.drop('diseases'))
    #feature=feature.reshape(48,48)
    #new_df = pd.DataFrame(data=None, columns=dataset.columns.drop(['diseases']))

    feature1 = feature1.append(pd.DataFrame([feature], columns=list(feature1)), ignore_index=False)
    prediction = model.predict(feature1)
    feature_names = feature1.columns[feature1.iloc[0] == 1].tolist() 

    disease= ()
    for i in prediction:
        disease = i
    symptoms = feature_names
    prognosis = classify_disease(symptoms, disease)

        
    feature2 = pd.DataFrame(data=[feature], columns=data.columns.drop('diseases'))
    feature1.rename(columns={'fever':'Fever', 'vomiting':'Vominiting', 'paralysis':'Paralysis',
                                 'reducedappetite':'Reduced Appetite' ,'coughing':'Coughing', 
                                 'dischargefromeyes':'Discharge from Eyes', 'hyperkeratosis':'Hyperkeratosis',
                                 'nasaldischarge':'Nasal Discharge','lethargy':'Lethargy', 'sneezing':'Sneezing',
                                 'diarrhea':'Diarrhea','depression':'Depression',
                                 'difficultyinbreathing':'Difficulty in Breathing','pain':'Pain','skinsores':'Skin Sores',
                                 'inflammation_eyes':'Inflammation Eyes','anorexia':'Anorexia','seizures':'Seizures',
                                 'dehydration':'Dehydration', 'weightloss':'Weightloss','bloodystool':'Bloody Stool',
                                 'weakness':'Weakness', 'inflammation_mouth':'Inflammation Eyes',
                                 'rapidheartbeat':'Rapid Heartbeat','fatigue':'Fatigue','swollenbelly':'Swollen Belly',
                                 'laziness':'Laziness','anemia':'Anemia','fainting':'Fainting',
                                 'reversesneezing':'Reverse Sneezing', 'gagging':'Gagging','lameness':'Lameness',
                                 'stiffness':'Stiffness', 'limping':'Limping', 'increasedthirst':'Increased Thirst',
                                 'increasedurination':'Increased Urination', 'excesssalivation':'Excess Salivation',
                                 'aggression':'Agression','foamingatmouth':'Foaming at Mouth',
                                 'difficultyinswallowing':'Difficulty in Swallowing', 'irritable':'Irritable',
                                 'pica':'Pica',
                                 'hydrophobia':'Hydrophobia','highlyexcitable':'Highly Excitable','shivering':'Shivering',
                                 'jaundice':'Jaundice', 'decreasedthirst':'Decreased Thirst',
                                 'decreasedurination':'Decreased Urination','bloodinurine':'Blood Urination',
                                 'palegums':'Pale Gums', 'ulcersinmouth':'Ulcers in Mouth','badbreath':'Bad Breath'},
                                  inplace=True)
    feature_names2 = feature1.columns[feature1.iloc[0] == 1].tolist()
    your_name = request.form.get('your_name')
    pet_name = request.form.get('pet_name')
#    return ('form.html', your_name==your_name, pet_name==pet_name)    
    if feature_names:

        return render_template('form.html', prediction_text='{}'.format(", ".join(prediction)),
                           feature_names='{}'.format(", ".join(feature_names2)),prognosis='{}'.format(prognosis),
                              yourtext='{}'.format(your_name) ,pet_text='{}'.format(pet_name))

    return render_template('form.html')
@app.route("/clear")    
def clear():
    predict.clear()
    #print("Data structure has been restarted.")


#@app.route('/get_names', methods=['POST' 'GET'])
#def get_names():
#    your_name = request.form['your_name']
#    pet_name = request.form['pet_name']
#    return redirect('/display_names')

@app.route('/display_names')
def display_names():
    return render_template('form.html', your_name==your_name, pet_name==pet_name)


## run
app.run(port=8000, debug=True, use_reloader=False)

