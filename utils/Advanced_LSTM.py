import torch
import torch.nn as nn

class AdvancedLSTM(nn.Module):

    def __init__(self,input_size,hidden_size,num_layers,telemetry_channels,columns_id):
        super().__init__()
        self.columns_index = columns_id
        self.networks = nn.ModuleDict()
        for c in telemetry_channels: 
            self.networks[c] = nn.ModuleDict({
                'lstm': nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True),
                'fc': nn.Linear(hidden_size, 1)
            })

    def forward(self,x):
            outputs = []
            ys_true = []
            for c in self.networks.keys():
                idx = self.columns_index[c]
                if idx == x.shape[2]:
                    x_mod = x[:, :, :idx]
                elif idx == 0:
                    x_mod = x[:, :, 1:]
                else:
                    x_mod = torch.cat((x[:, :, :idx], x[:, :, idx+1:]), dim=2)


                out, _ = self.networks[c]['lstm'](x_mod)

                outputs.append(self.networks[c]['fc'](out).squeeze(-1))
                ys_true.append(x[:,:,idx])

            return outputs, ys_true