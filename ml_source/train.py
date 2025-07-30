import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib 

df = pd.read_csv("C:\\Users\\LENOVO\\OneDrive\\Desktop\\demo_covid\\pneumonia_covid_diagnosis_dataset.csv")
df = df.drop("Is_Curable", axis=1)
columns=['Gender', 'Fever', 'Cough', 'Fatigue', 'Breathlessness', 'Comorbidity', 'Stage', 'Type']
le=LabelEncoder()
for col in columns:
    le=LabelEncoder()
    df[col]=le.fit_transform(df[col])
    
    X = df.drop("Survival_Rate", axis=1)
    y = df["Survival_Rate"]
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestRegressor()
model.fit(X_train, y_train)
pred = model.predict(X_test)

joblib.dump(model,"covid_diag.pkl")