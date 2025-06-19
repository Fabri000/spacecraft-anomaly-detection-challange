import torch
import numpy as np
import torch.nn as nn
from tqdm.auto import tqdm

import gc

class AnomalyScoreEstimator:
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
                scores[telemetry] = torch.sigmoid(self.single_telemetry_score_estimation(telemetries_errors[telemetry]))
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        return scores

    def calculate_epsilon(self, means:torch.tensor, stds:torch.tensor ,z:int=3):
        return means + z * stds

    def single_telemetry_score_estimation(self, serie:torch.tensor ,h:int=64):
        scores = []
        padded_serie = torch.cat([torch.zeros(serie.shape[0],h-1), serie], dim=1)
        windows = padded_serie.unfold(dimension=1, size=h, step=1)
        
        means = torch.mean(windows,dim=2).squeeze(0)
        stds = torch.std(windows,dim=2).squeeze(0)

        epsilons = self.calculate_vectorized_epsilon(windows, means,stds)

        denominators = means + stds
        denominators = torch.where(denominators == 0, torch.tensor(1e-8), denominators)

        scores = (torch.max(windows,dim=2)[0].squeeze(0) - epsilons) / denominators    

        return scores
    
    def calculate_vectorized_epsilon(self,windows:torch.tensor, means:torch.tensor, stds:torch.tensor, init_percentile:float = 95.0, iters: int = 50,step: float = 5e-2):
        
        epsilons = torch.quantile(windows, init_percentile / 100.0, dim=2)
        best_values = torch.zeros(epsilons.shape[0],epsilons.shape[1])

        for _ in range(iters):
            steps = torch.rand(epsilons.shape[0],epsilons.shape[1]) * (-step) + step
            candidate_epsilons = epsilons + steps

            normal_mask = windows < candidate_epsilons.unsqueeze(-1)
            normal_samples = torch.where(normal_mask, windows,torch.zeros_like(windows))
            
            delta_means = means - torch.mean(normal_samples,dim=2)
            delta_stds = stds - torch.std(normal_samples,dim=2)

            E_seq = self.efficient_calculate_continouse_seqs(windows, ~normal_mask)

            tmp_values = ((delta_means / means) + (delta_stds / stds)) / (torch.sum(~normal_mask) + E_seq ** 2)

            improved_mask  = tmp_values >= best_values
            best_values = torch.where(improved_mask, tmp_values, best_values)
            epsilons = torch.where(improved_mask, candidate_epsilons, epsilons)

        return epsilons

    def efficient_calculate_continouse_seqs(self,windows:torch.tensor,anomaly_masks:torch.tensor):
        num_windows = windows.shape[1]
        padded = torch.cat([
            torch.zeros(num_windows, 1, dtype=torch.bool),  # Left padding
            anomaly_masks.squeeze(0),
            torch.zeros(num_windows, 1, dtype=torch.bool)   # Right padding
        ], dim=1)

        diff = padded[:, 1:].int() - padded[:, :-1].int()
        sequence_starts = (diff == 1).sum(dim=1)

        return sequence_starts
