import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm

from itertools import groupby

import gc

class AnomalyDetector:
    def __init__(self,model:nn.Module):
        self.telemetry_estimator = model
        self.telemetry_channels = list(model.networks.keys())

    def calculate_errors(self, time_series:torch.tensor):
        telemetry_predictions, telemetry_true = self.telemetry_estimator(time_series)
        errors = {}
        for i in range(len(telemetry_predictions)):
            errors[self.telemetry_channels[i]] = self.calculate_telemetry_errors(telemetry_predictions[i],telemetry_true[i])
        return errors

    def calculate_telemetry_errors(self,time_serie_prediction:torch.tensor, time_serie_true:torch.tensor):
        return torch.abs(time_serie_true - time_serie_prediction)
    
    def emwa(self,telemetries_errors:dict):
        for telemetry in telemetries_errors.keys():
            telemetries_errors[telemetry] = self.emwa_telemetry(telemetries_errors[telemetry])

    def emwa_telemetry(self,telemetry_errors:torch.tensor, alpha:float = 0.3):
        for t in range(1,len(telemetry_errors)):
            telemetry_errors[t] = telemetry_errors[t] * alpha  + (1 - alpha) * telemetry_errors[t-1]
            
        return telemetry_errors
    
    def estimate_scores(self, telemetries_errors:dict):
        scores = {}
        for telemetry in tqdm(telemetries_errors.keys()):

            with torch.no_grad():
                scores[telemetry] = self.single_telemetry_score_estimation(telemetries_errors[telemetry])
            
            del errors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        return scores


    def single_telemetry_score_estimation(self, serie:torch.tensor ,h:int=64):
        scores = []
        for t in range(serie.shape[1]):
            start_idx = max(0, t - h)
            end_idx = t + 1
            window = serie[:, start_idx:end_idx].flatten()
            epsilon = 0

            if window.numel() == 1:
                mean = window[0].item()
                std = 0
            else:
                mean = torch.mean(window).item()
                std = torch.std(window).item()
            
            epsilon = self.calculate_epsilon(mean,std)
            
            scores.append((torch.max(window).item() - epsilon) / (mean + std))

            if t % 100 == 0:
                del window
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()

        return scores
    
    

    def calculate_epsilon(self, mean:float,std:float,z:int=5):
        return mean + z * std

    def calculate_approximate_epsilon(self, sequence:torch.tensor, mean:float, std:float , step:float=1e-2, iters:int=15):
        
        best_epsilon = 1
        best_value = 0

        epsilon =  np.random.uniform(0,1)
        value = 0

        for _ in range(iters):
            candidate_epsilon = np.clip(epsilon + np.random.uniform(-step,step), 0, 1)

            normal = sequence < candidate_epsilon

            E_seq = self.calculate_continouse_seq(sequence,~normal)

            c_e_a = sequence[normal]
            delta_mean = mean - torch.mean(c_e_a)
            delta_std = 0
            if c_e_a.shape[0] <= 1:
                delta_std = std
            else:
                delta_std = std - torch.std(c_e_a)

            tmp_value = ((delta_mean / mean) + (delta_std / std)) / (torch.sum(~normal)+ E_seq ** 2)

            if tmp_value > value:
                value = tmp_value
                epsilon = candidate_epsilon

            if tmp_value > best_value:
                best_value = value
                best_epsilon  = epsilon
        
        return best_epsilon


    def calculate_continouse_seq(self,sequence,anomaly):
        sequences = []
        print(anomaly)
        for is_anomaly, group in groupby(zip(anomaly, sequence ), key=lambda x: x[1]):
            if is_anomaly:
                sequences.append(list(group))

        return len(sequences)