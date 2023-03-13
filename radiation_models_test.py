# Dakota Kosiorek
# Goal: Find the smallest amount of lead needed to block 1-10 mega electron volts (MeV)

import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle


mpl.rcParams['figure.figsize'] = (11, 6)
mpl.rcParams['axes.grid'] = False

class Mev_Shield():
    def __init__(self, material, MeV, thickness_cm):
        self.material = material
        self.MeV = MeV
        self.thickness_cm = thickness_cm

def main():
    shields = list()
    
    for material in os.listdir("models"):
        material_pth = os.path.join("models", material)
        low_model = pickle.load(open(os.path.join(material_pth, "low.pickle"), "rb"))
        high_model = pickle.load(open(os.path.join(material_pth, "high.pickle"), "rb"))
        
        predictor_values = [1]
        predictor_values_low = sorted(i for i in predictor_values if i < 20)
        predictor_values_high = sorted(i for i in predictor_values if i >= 20)
        
        low_Model_prediction = list()
        high_model_predictions = list()
        
        if len(predictor_values_low) > 0:
            low_Model_prediction = low_model.predict(pd.DataFrame(predictor_values_low, columns=['Kinetic Energy MeV']))
            for (i, val) in enumerate(predictor_values_low):
                shields.append(Mev_Shield(material=material, MeV=val, thickness_cm=low_Model_prediction[i]))
        if len(predictor_values_high) > 0:
            high_model_predictions = high_model.predict(pd.DataFrame(predictor_values_high, columns=['Kinetic Energy MeV']))
            for (i, val) in enumerate(predictor_values_high):
                shields.append(Mev_Shield(material=material, MeV=val, thickness_cm=high_model_predictions[i]))
        
    shields.sort(key=lambda x: x.thickness_cm, reverse=True)
    df = pd.DataFrame([b.__dict__ for b in shields])
    df = df.rename({'material': 'Material', 'MeV': 'MeV', 'thickness_cm': 'Thickness (cm)'}, axis=1)
    
    #df.to_csv("output.csv", index=False)
    print(df.loc[df['Material'] == "Lead"])

if __name__ == "__main__":
    print("Starting 'radiation_test.py'...\n")
    main()
    print("\n'radiation_test.py' complete!")