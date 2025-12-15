import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class RecursiveCoherenceTransformer(nn.Module):
    def __init__(self, vocab_size=10000, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, max_iter=5, alpha=0.5, rank=16, epsilon=1e-3, lambda_moco=0.1, lambda_mono=0.1, lambda_cptp=0.1, rc_mode='outer'):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.alpha = alpha
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.lambda_moco = lambda_moco
        self.lambda_mono = lambda_mono
        self.lambda_cptp = lambda_cptp
        self.rc_mode = rc_mode
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 512, d_model))
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward) for _ in range(num_layers)
        ])
        
        self.P_o = nn.Linear(d_model, rank)
        self.P_s = nn.Linear(d_model, rank)
        
        self.num_kraus = 4
        self.kraus_mlps = nn.ModuleList([nn.Sequential(
            nn.Linear(rank + rank, rank * rank),  # U + h_proj
            nn.ReLU(),
            nn.Linear(rank * rank, rank * rank)
        ) for _ in range(self.num_kraus)])
        
        self.halt_head = nn.Sequential(
            nn.Linear(rank + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.h_to_rank = nn.Linear(d_model, rank)  # Learned proj for h
        
        self.W_r = nn.Parameter(torch.randn(rank * rank, rank))
        self.V = nn.Parameter(torch.randn(rank, rank))
        
        self.readout = nn.Linear(rank, vocab_size)
    
    def normalize_factors(self, U):
        norm = torch.norm(U, dim=1, keepdim=True).clamp(min=1e-6)
        return U / norm
    
    def von_neumann_entropy_low_rank(self, U, eps=1e-8):
        _, S, _ = torch.svd(U)
        S = S.clamp(min=eps)
        return - (S * torch.log(S)).sum(1)
    
    def relative_coherence_low_rank(self, U):
        U_sq = U**2
        diag_S = - (U_sq * torch.log(U_sq.clamp(min=1e-8))).sum(1)  # Rough approx
        S = self.von_neumann_entropy_low_rank(U)
        return diag_S - S
    
    def apply_kraus_update_low_rank(self, U, h):
        h_proj = self.h_to_rank(h)
        input_k = torch.cat([U, h_proj], dim=1)
        
        sum_KdagK = torch.zeros(U.shape[0], self.rank, self.rank, device=U.device)
        new_U = torch.zeros_like(U)
        
        for mlp in self.kraus_mlps:
            K_flat = mlp(input_k)  # [b, r*r]
            K = K_flat.reshape(U.shape[0], self.rank, self.rank)
            Kdag = K.transpose(1,2)  # Assume real
            sum_KdagK += torch.bmm(Kdag, K)
            
            new_U += torch.bmm(K, U.unsqueeze(2)).squeeze(2)
        
        new_U = self.normalize_factors(new_U)
        
        return new_U, sum_KdagK
    
    def rc_loop(self, h_or_pool, training=False):
        psi_o = self.normalize_factors(self.P_o(h_or_pool))
        psi_s = self.normalize_factors(self.P_s(h_or_pool))
        
        U = self.alpha * psi_o + (1 - self.alpha) * psi_s
        U = self.normalize_factors(U)
        
        S_list = [self.von_neumann_entropy_low_rank(U)]
        C_list = [self.relative_coherence_low_rank(U)]
        cptp_terms = []
        
        prev_S = S_list[-1]
        prev_C = C_list[-1]
        
        for iter in range(self.max_iter):
            U, sum_KdagK = self.apply_kraus_update_low_rank(U, h_or_pool)
            if training:
                I = torch.eye(self.rank, device=U.device).unsqueeze(0).repeat(U.shape[0], 1, 1)
                cptp_term = torch.norm(sum_KdagK - I, dim=(1,2)).mean()
                cptp_terms.append(cptp_term)
            
            S = self.von_neumann_entropy_low_rank(U)
            C = self.relative_coherence_low_rank(U)
            S_list.append(S)
            C_list.append(C)
            
            delta_S = torch.abs(S - prev_S)
            if torch.all(delta_S < self.epsilon):
                break
            
            halt_input = torch.cat([U, S.unsqueeze(1), C.unsqueeze(1)], dim=1)
            halt_prob = self.halt_head(halt_input)
            if torch.mean(halt_prob) > 0.9:
                break
            
            prev_S = S
            prev_C = C
        
        if training:
            cptp_loss = sum(cptp_terms) / len(cptp_terms) if cptp_terms else torch.tensor(0.0)
        else:
            cptp_loss = None
        
        return U, S_list, C_list, cptp_loss
    
    def forward(self, input_ids, targets=None, training=False):
        batch, seq = input_ids.shape
        x = self.embedding(input_ids) + self.pos_encoder[:, :seq, :]
        
        S_lists = []  # Collect for interleaved
        C_lists = []
        cptp_losses = []
        
        if self.rc_mode == 'interleaved':
            for layer in self.transformer_layers:
                x = layer(x)
                pool = x.mean(1)
                U, S_list, C_list, cptp_loss = self.rc_loop(pool, training)
                S_lists.append(S_list)
                C_lists.append(C_list)
                if training and cptp_loss is not None:
                    cptp_losses.append(cptp_loss)
                x[:, -1, :] += F.linear(U, torch.randn(self.rank, self.d_model))  # Temp random; ideally learned
            h = x[:, -1, :]
        
        elif self.rc_mode == 'head':
            for layer in self.transformer_layers:
                x = layer(x)
            pool = x.mean(1)
            U, S_list, C_list, cptp_loss = self.rc_loop(pool, training)
            S_lists.append(S_list)
            C_lists.append(C_list)
            if training and cptp_loss is not None:
                cptp_losses.append(cptp_loss)
            h = pool
        
        else:  # outer
            for layer in self.transformer_layers:
                x = layer(x)
            h = x[:, -1, :]
            U, S_list, C_list, cptp_loss = self.rc_loop(h, training)
            S_lists.append(S_list)
            C_lists.append(C_list)
            if training and cptp_loss is not None:
                cptp_losses.append(cptp_loss)
        
        h_rank = self.h_to_rank(h)
        U_flat = U.reshape(batch, -1)
        u = torch.matmul(U_flat, self.W_r) + torch.matmul(h_rank, self.V)
        logits = self.readout(u)
        
        if not training:
            return logits
        
        # For interleaved/head, average lists/losses; for simplicity, take last
        S_list = S_lists[-1] if S_lists else []
        C_list = C_lists[-1] if C_lists else []
        total_cptp = sum(cptp_losses) / len(cptp_losses) if cptp_losses else torch.tensor(0.0)
        
        return logits, S_list, C_list, total_cptp
    
    def compute_loss(self, logits, targets, S_list, C_list, cptp_loss):
        nll = F.cross_entropy(logits, targets)
        
        delta_S = torch.diff(torch.stack(S_list), dim=0)
        l_mono = torch.mean(F.relu(delta_S))
        
        delta_C = torch.diff(torch.stack(C_list), dim=0)
        l_moco = torch.mean(F.relu(-delta_C))
        
        total_loss = nll + self.lambda_mono * l_mono + self.lambda_moco * l_moco + self.lambda_cptp * cptp_loss
        return total_loss
    
    def generate(self, input_ids, max_length=20, temperature=1.0):
        generated = input_ids.clone()
        for _ in range(max_length):
            logits = self(generated)
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == 0:
                break
        return generated

# Example
if __name__ == "__main__":
    model = RecursiveCoherenceTransformer()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    batch_size = 4
    seq_len = 10
    for epoch in range(5):
        input_ids = torch.randint(0, 10000, (batch_size, seq_len))
        targets = torch.randint(0, 10000, (batch_size,))
        
        model.train()
        logits, S_list, C_list, cptp_loss = model(input_ids, targets, training=True)
        loss = model.compute_loss(logits, targets, S_list, C_list, cptp_loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}: Loss {loss.item()}")
    
    input_ids = torch.randint(0, 10000, (1, 5))
    gen = model.generate(input_ids)
    print("Generated:", gen)