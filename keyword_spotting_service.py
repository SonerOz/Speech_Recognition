import tensorflow.keras as keras 
import numpy as np
import librosa
import sys

sys.setrecursionlimit(1500)

MODEL = "model.h5"
Considerable_Samples = 22050

class _Keyword_Spotting_Service:
    
    model = None
    mapping = [
        "dataset\\bird",
        "dataset\\cat",
        "dataset\\dog",
        "dataset\\happy",
        "dataset\\house",
        "dataset\\wow"
    ]
    
    instance = None
    
    def predict(self, file_path):
        
      
        MFCCs = self.preprocess(file_path) 
        
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self.mapping[predicted_index]
        
        return predicted_keyword

    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):
        
        signal= librosa.load(file_path)
    
        if len(signal) > Considerable_Samples:
            signal = signal[:Considerable_Samples]
      
        MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        
        return MFCCs.T
    
    
def Keyword_Spotting_Service():
  
    if _Keyword_Spotting_Service.instance is None:
        _Keyword_Spotting_Service.instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL)
    return _Keyword_Spotting_Service.instance
    
if __name__ == "__main__":
    kss = Keyword_Spotting_Service()
    
    keyword1 = kss.predict("test\\bird.wav")
    keyword2 = kss.predict("test\\cat.wav")
    
    print(f"Predicted keywords: {keyword1}, {keyword2}")