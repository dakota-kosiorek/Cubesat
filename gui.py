import tkinter as tk
import tkinter.ttk as ttk
import pandas as pd
import os
import pickle

def main():
    # Create window
    window = tk.Tk()
    window.geometry("800x400")
    window.winfo_toplevel().title("SPEX CubeSat Proton Shielding Material Calculator (0-10,000 MeV)")
    
    frame1 = tk.Frame(master=window, height=50)
    frame2 = tk.Frame(master=window, height=200)
    
    output_text = tk.Text(
        master=frame2
    )
    
    warning_label = tk.Label(
        master=frame1,
        text="MeV needs to be a number!",
        bg="white",
        fg="red"
    )
    
    MeV_label = tk.Label(
        master=frame1,
        text="MeV:",
        fg="black"
    )
    
    MeV_entry = tk.Entry(
        master=frame1,
        bg="white",
        fg="black", 
        width=25
    )
    
    button = tk.Button(
        master=frame1,
        text="Calculate",
        width=15,
        height=1,
        bg="black",
        fg="white",
        command=lambda: calculate(warning_label=warning_label, output_text=output_text, MeV=MeV_entry.get())
    )
    
    frame1.pack(fill=tk.BOTH, side=tk.TOP)
    frame2.pack(fill=tk.BOTH, side=tk.TOP)
    MeV_label.grid(row=1, column=1)
    MeV_entry.grid(row=1, column=2)
    button.grid(row=2, column=2)
    
    window.mainloop()

class Mev_Shield():
    def __init__(self, material, MeV, thickness_cm):
        self.material = material
        self.MeV = MeV
        self.thickness_cm = thickness_cm

def calculate(warning_label, output_text, MeV):
    try:
        MeV = float(MeV)
        warning_label.grid_forget()
        output_text.delete("1.0","end")
        
        shields = list()
    
        for material in os.listdir("models"):
            material_pth = os.path.join("models", material)
            low_model = pickle.load(open(os.path.join(material_pth, "low.pickle"), "rb"))
            high_model = pickle.load(open(os.path.join(material_pth, "high.pickle"), "rb"))
            
            predictor_values = list()
            predictor_values.append(MeV)
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
        df.to_csv("output.csv", index=False)

        output_text.insert(tk.END, str(df))
        output_text.grid(row=2, column=1)
    
    except:
        warning_label.grid(row=1, column=3)
        output_text.grid_forget()

if __name__ == "__main__":
    main()