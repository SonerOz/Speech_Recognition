import librosa
import os
import json
import numpy as np 

DATASET = "dataset"
JSON = "data.json"
Considerable_Samples = 22050 

def prepare_dataset(datase_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):
    
    data = {
        "mappings": [],
        "labels" : [],
        "MFCCs" : [],
        "files" : []
    }
    
    for i, (dirpath, filenames) in enumerate(os.walk(datase_path)):
        
        if dirpath is not datase_path:
            
           
            category = dirpath.split("/")[-1] 
            data["mappings"].append(category)
            print(f"Processing {category}")
            
            
            for f in filenames:
                
               
                file_path = os.path.join(dirpath, f)
                
                
                signal, sample_rate = librosa.load(file_path)
                
               
                if len(signal) >= Considerable_Samples:
                    
                    
                    signal = signal[:Considerable_Samples]
                    
                    
                    MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
                    
                   
                    data["labels"].append(i-1)
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["files"].append(file_path)
                    print(f"{file_path}: {i-1}")
    
   
    
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    prepare_dataset(DATASET, JSON)