import torch
from typing import List
import torch.nn.functional as F
import numpy as np

# [참고] lip은 lipschitz를 의미합니다.
class LiDER:
    def __init__(self, model, device, forward_flops, backward_flops):
        self.all_lips = []
        self.model = model
        self.device = device
        self.forward_flops = forward_flops
        self.backward_flops = backward_flops
        self.total_flops = 0
        self.alpha = .3
        self.beta = .1
        self.len_stream_features = 0
    
    def update_lip_values(self, stream_x, stream_y):
        self.model.eval()
        _, stream_features = self.model(stream_x, get_features=True, include_out_for_features=True)
        self.total_flops += len(stream_x) * self.forward_flops
        lip_inputs = [stream_x] + stream_features[:-1]

        lip_values = self.get_feature_lip_coeffs(lip_inputs)
        lip_values = torch.stack(lip_values, dim=1)
        lip_values = lip_values.detach().cpu()

        self.all_lips.append(lip_values)
        
        del lip_values
        torch.cuda.empty_cache()
        
        if self.len_stream_features == 0: self.len_stream_features = len(stream_features)

        self.model.train()

    def reset_lip_values(self):
        self.budget_lip = torch.cat(self.all_lips, dim=0).mean(0)
        self.budget_lip = self.budget_lip.to(self.device)
        self.model.lip_coeffs = torch.autograd.Variable(torch.randn(self.len_stream_features-1, dtype=torch.float), requires_grad=True)
        self.model.lip_coeffs.data = self.budget_lip.detach().clone()

        self.model.train()

    def get_feature_lip_coeffs(self, features: List[torch.Tensor], create_att_map=False) -> List[torch.Tensor]:
        N = len(features)-1
        B = len(features[0])
        lip_values = [torch.zeros(B, device=self.device, dtype=features[0].dtype)]*N

        for i in range(N):
            fma,fmb = features[i], features[i+1]
            fmb = F.adaptive_avg_pool1d(fmb.reshape(*fmb.shape[:2],-1).permute(0,2,1), fma.shape[1]).permute(0,2,1).reshape(fmb.shape[0],-1,*fmb.shape[2:])
            L = self.get_layer_lip_coeffs(fma, fmb)
            L = L.reshape(B)
            lip_values[i] = L if not create_att_map else torch.sigmoid(L)
        return lip_values
    
    # beta
    def buffer_lip_loss(self, features: List[torch.Tensor]) -> torch.Tensor:
        lip_values = self.get_feature_lip_coeffs(features)
        # (B, F)
        lip_values = torch.stack(lip_values, dim=1)
        print("lip_loss,", lip_values.mean() * self.beta)
        return lip_values.mean() * self.beta

    # alpha
    def budget_lip_loss(self, features: List[torch.Tensor]) -> torch.Tensor:
        loss = 0
        lip_values = self.get_feature_lip_coeffs(features)
        # (B, F)
        lip_values = torch.stack(lip_values, dim=1)

        tgt = F.relu(self.model.lip_coeffs[:len(lip_values[0])])
        tgt = tgt.unsqueeze(0).expand(lip_values.shape)
        # device of lip_values
        loss += F.l1_loss(lip_values, tgt)
        print("budget_loss,", loss)
        return loss * self.alpha

    def budget_loss(self, features: List[torch.Tensor]) -> torch.Tensor:
        lip_values = self.get_feature_lip_coeffs(features)
        lip_values = torch.stack(lip_values, dim=1)
        
        tgt = F.relu(self.model.lip_coeffs[:len(lip_values[0])])
        tgt = tgt.unsqueeze(0).expand(lip_values.shape)
        
        return F.l1_loss(lip_values, tgt) * self.alpha + lip_values.mean() * self.beta
    
    def get_norm(self, t: torch.Tensor):
        return torch.norm(t, dim=1, keepdim=True)+torch.finfo(torch.float32).eps
    
    def transmitting_matrix(self, fm1: torch.Tensor, fm2: torch.Tensor):
        if fm1.size(2) > fm2.size(2):
            fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(-2), fm2.size(-1)))

        fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)
        fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1, 2)
        b, n, p = fm1.shape
        b, p, m = fm2.shape
        self.total_flops += (b*n*m*(2*p-1)) / 10e9
        fsp = torch.bmm(fm1, fm2) / fm1.size(2)
        return fsp

    def compute_transition_matrix(self, front: torch.Tensor, latter: torch.Tensor):
        # matrix product between (n×p) and  (p×m)
        # nm(2p−1)
        transition_matrix = self.transmitting_matrix(front, latter)
        b, n, p = transition_matrix.shape
        self.total_flops += (b * n*n * (2*p-1)) / 10e9
        return torch.bmm(transition_matrix, transition_matrix.transpose(2,1))

    # getting top eigenvalue of transition matrix
    def top_eigenvalue(self, transition_matrix: torch.Tensor, n_power_iterations=10):
        start_grad_it = 2

        v = torch.rand(transition_matrix.shape[0], transition_matrix.shape[1], 1).to(transition_matrix.device, dtype=transition_matrix.dtype)
        for itt in range(n_power_iterations):
            with torch.set_grad_enabled(itt>=start_grad_it):
                self.total_flops += (transition_matrix.shape[0] * transition_matrix.shape[1] * v.shape[2] * (2*transition_matrix.shape[2] - 1)) / 10e9
                m = torch.bmm(transition_matrix, v)
                n = (torch.norm(m, dim=1).unsqueeze(1) + torch.finfo(torch.float32).eps)
                v = m / n

        top_eigenvalue = torch.sqrt(n / (torch.norm(v, dim=1).unsqueeze(1) + torch.finfo(torch.float32).eps))
        return top_eigenvalue

    def get_single_feature_lip_coeffs(self, feature: torch.Tensor) -> torch.Tensor:
        stream_batch_size = len(feature[0])
        permute=torch.from_numpy(np.random.permutation(stream_batch_size)).to(self.device, dtype=torch.int64)
        features_a, permuted_feature = feature, feature[permute] 

        features_a, permuted_feature = features_a.double(), permuted_feature.double()
        features_a, permuted_feature = features_a / self.get_norm(features_a), permuted_feature / self.get_norm(permuted_feature)

        transition_matrix = self.compute_transition_matrix(features_a, permuted_feature)
        top_eigenvalue = self.top_eigenvalue(K=transition_matrix)
        return top_eigenvalue

    def get_layer_lip_coeffs(self, features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
        features_a, features_b = features_a.double(), features_b.double()
        features_a, features_b = features_a / self.get_norm(features_a), features_b / self.get_norm(features_b)
        self.total_flops += (features_a.shape.numel() + features_b.shape.numel()) / 10e9

        transition_matrix = self.compute_transition_matrix(features_a, features_b)
        top_eigenvalue = self.top_eigenvalue(transition_matrix=transition_matrix)
        return top_eigenvalue