from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle


app = Flask(__name__)

sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv("datasets/medications.csv")
diets = pd.read_csv("datasets/diets.csv")



svc = pickle.load(open("models/svc.pkl",'rb'))




def helper(des):
    descr = description[description['Disease'] == des]['Description']
    descr = " ".join(w for w in descr)

    prec = precautions[precautions['Disease'] == des][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    prec = [col for col in prec.values]

    med = medications[medications['Disease'] == des]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == des]['Diet']
    die = [die for die in die.values]

    wrk = workout[workout['disease'] == des]['workout']

    return descr, prec, med, die, wrk


# function to predict disease
disease_dict = {0: '(vertigo) Paroymsal  Positional Vertigo',
                1: 'AIDS',
                2: 'Acne',
                3: 'Alcoholic hepatitis',
                4: 'Allergy',
                5: 'Arthritis',
                6: 'Bronchial Asthma',
                7: 'Cervical spondylosis',
                8: 'Chicken pox',
                9: 'Chronic cholestasis',
                10: 'Common Cold',
                11: 'Dengue',
                12: 'Diabetes ',
                13: 'Dimorphic hemmorhoids(piles)',
                14: 'Drug Reaction',
                15: 'Fungal infection',
                16: 'GERD',
                17: 'Gastroenteritis',
                18: 'Heart attack',
                19: 'Hepatitis B',
                20: 'Hepatitis C',
                21: 'Hepatitis D',
                22: 'Hepatitis E',
                23: 'Hypertension ',
                24: 'Hyperthyroidism',
                25: 'Hypoglycemia',
                26: 'Hypothyroidism',
                27: 'Impetigo',
                28: 'Jaundice',
                29: 'Malaria',
                30: 'Migraine',
                31: 'Osteoarthristis',
                32: 'Paralysis (brain hemorrhage)',
                33: 'Peptic ulcer diseae',
                34: 'Pneumonia',
                35: 'Psoriasis',
                36: 'Tuberculosis',
                37: 'Typhoid',
                38: 'Urinary tract infection',
                39: 'Varicose veins',
                40: 'hepatitis A'}


symptoms_dict = {
    "itching": 1,
    "skin_rash": 3,
    "nodal_skin_eruptions": 4,
    "continuous_sneezing": 4,
    "shivering": 5,
    "chills": 3,
    "joint_pain": 3,
    "stomach_pain": 5,
    "acidity": 3,
    "ulcers_on_tongue": 4,
    "muscle_wasting": 3,
    "vomiting": 5,
    "burning_micturition": 6,
    "spotting_urination": 6,
    "fatigue": 4,
    "weight_gain": 3,
    "anxiety": 4,
    "cold_hands_and_feets": 5,
    "mood_swings": 3,
    "weight_loss": 3,
    "restlessness": 5,
    "lethargy": 2,
    "patches_in_throat": 6,
    "irregular_sugar_level": 5,
    "cough": 4,
    "high_fever": 7,
    "sunken_eyes": 3,
    "breathlessness": 4,
    "sweating": 3,
    "dehydration": 4,
    "indigestion": 5,
    "headache": 3,
    "yellowish_skin": 3,
    "dark_urine": 4,
    "nausea": 5,
    "loss_of_appetite": 4,
    "pain_behind_the_eyes": 4,
    "back_pain": 3,
    "constipation": 4,
    "abdominal_pain": 4,
    "diarrhoea": 6,
    "mild_fever": 5,
    "yellow_urine": 4,
    "yellowing_of_eyes": 4,
    "acute_liver_failure": 6,
    "fluid_overload": 6,
    "swelling_of_stomach": 7,
    "swelled_lymph_nodes": 6,
    "malaise": 6,
    "blurred_and_distorted_vision": 5,
    "phlegm": 5,
    "throat_irritation": 4,
    "redness_of_eyes": 5,
    "sinus_pressure": 4,
    "runny_nose": 5,
    "congestion": 5,
    "chest_pain": 7,
    "weakness_in_limbs": 7,
    "fast_heart_rate": 5,
    "pain_during_bowel_movements": 5,
    "pain_in_anal_region": 6,
    "bloody_stool": 5,
    "irritation_in_anus": 6,
    "neck_pain": 5,
    "dizziness": 4,
    "cramps": 4,
    "bruising": 4,
    "obesity": 4,
    "swollen_legs": 5,
    "swollen_blood_vessels": 5,
    "puffy_face_and_eyes": 5,
    "enlarged_thyroid": 6,
    "brittle_nails": 5,
    "swollen_extremeties": 5,
    "excessive_hunger": 4,
    "extra_marital_contacts": 5,
    "drying_and_tingling_lips": 4,
    "slurred_speech": 4,
    "knee_pain": 3,
    "hip_joint_pain": 2,
    "muscle_weakness": 2,
    "stiff_neck": 4,
    "swelling_joints": 5,
    "movement_stiffness": 5,
    "spinning_movements": 6,
    "loss_of_balance": 4,
    "unsteadiness": 4,
    "weakness_of_one_body_side": 4,
    "loss_of_smell": 3,
    "bladder_discomfort": 4,
    "foul_smell_ofurine": 5,
    "continuous_feel_of_urine": 6,
    "passage_of_gases": 5,
    "internal_itching": 4,
    "toxic_look_(typhos)": 5,
    "depression": 3,
    "irritability": 2,
    "muscle_pain": 2,
    "altered_sensorium": 2,
    "red_spots_over_body": 3,
    "belly_pain": 4,
    "abnormal_menstruation": 6,
    "dischromic_patches": 6,
    "watering_from_eyes": 4,
    "increased_appetite": 5,
    "polyuria": 4,
    "family_history": 5,
    "mucoid_sputum": 4,
    "rusty_sputum": 4,
    "lack_of_concentration": 3,
    "visual_disturbances": 3,
    "receiving_blood_transfusion": 5,
    "receiving_unsterile_injections": 2,
    "coma": 7,
    "stomach_bleeding": 6,
    "distention_of_abdomen": 4,
    "history_of_alcohol_consumption": 5,
    "fluid_overload": 4,
    "blood_in_sputum": 5,
    "prominent_veins_on_calf": 6,
    "palpitations": 4,
    "painful_walking": 2,
    "pus_filled_pimples": 2,
    "blackheads": 2,
    "scurring": 2,
    "skin_peeling": 3,
    "silver_like_dusting": 2,
    "small_dents_in_nails": 2,
    "inflammatory_nails": 2,
    "blister": 4,
    "red_sore_around_nose": 2,
    "yellow_crust_ooze": 3,
    "prognosis": 5
}


def predict_disease(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))

    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return disease_dict[svc.predict([input_vector])[0]]





@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method=='POST':
        symptoms = request.form.get('symptoms')
        print(symptoms)
        if symptoms == "Symptoms":
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template('index.html', message=message)
        else:

            user_symptom = [s.strip() for s in symptoms.split(',')]
            user_symptom = [sym.strip("[]' ") for sym in user_symptom]
            predicted_disease = predict_disease(user_symptom)

            descr, prec, med, die, wrk = helper(predicted_disease)

            my_precautions = []
            for i in prec[0]:
                my_precautions.append(i)

            return render_template('index.html', predicted_disease=predicted_disease, dis_des=descr,
                                   my_precautions=prec, medications=med, my_diet=die,
                                   workout=wrk)

        return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/blog')
def blog():
    return render_template('blog.html')

@app.route('/developer')
def developer():
    return render_template('developer.html')


#python main
if __name__ == '__main__':
    app.run(debug=True)
