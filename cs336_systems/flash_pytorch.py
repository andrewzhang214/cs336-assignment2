
import math

from einops import einsum, reduce
import torch




class FlashAttention2Pytorch(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal=False):
        
        device = q.device

        # Initialize tile sizes
        Bq = Bk = 16 # Tile size


        # Create tiles to iterate over
        batch, seq_len, d = q.shape
        Tq = math.ceil(seq_len / Bq) # Query tiles to iterate over
        Tk = math.ceil(seq_len / Bk) # Key and value tiles to iterate over
        scale = math.sqrt(d)

        # Initialize output O (batch, seq_len, d)
        output = torch.empty(batch, seq_len, d, device=device, dtype=torch.float32)
        logsumexp = torch.empty(batch, seq_len, device=device, dtype=torch.float32)

        for i in range(1, Tq+1): # Iterates over sets of query vectors
            # Load Qi from VRAM (batch, Bq, d)
            q_tile = q[:, Bq, :]

            # Initialize Oi_0 (batch, Bq, d) 
            o_tile = torch.zeros(batch, Bq, d, device=device, dtype=torch.float32)

            # Initialize li (batch, Bq)  Running sum of exponentials per row (Changes as we go across seq)
            l = torch.zeros(batch, Bq, device=device, dtype=torch.float32)

            # Initialize mi (batch, Bq)  Running max per row (Changes as we go across seq)
            m = torch.zeros(batch, Bq, device=device, dtype=torch.float32)

            for j in range(1, Tk+1): # Iterates over sets of key and value vectors and now iteratively craetes O horizontally across the seq
                # Load Kj and Vj (batch, Bk, d)
                k_tile = k[:, j*Bk:(j+1)*Bk, :]
                v_tile = v[:, j*Bk:(j+1)*Bk, :]

                # Compute S: (batch, Bq, Bd) Representing the scores for this tile
                # This is EXACT since we're loading the whole row and column (K and Q) per tile
                s = einsum(q_tile, k_tile, "... Bq d, ... Bk d -> ... Bq Bk") / scale


                # Compute m: Running max per row (batch, Bq)
                s_row_max = reduce(s, "... Bq Bk -> ... Bq", "max")
                m_prev = m
                m = torch.maximum(s_row_max, m) 

                # Compute partial P (batch, Bq, Bk)
                # This is NOT EXACT since the running max is changing (also it's not normalized yet)
                p_partial = torch.exp(s - m) # Verify if it broadcasts, otherwise need to expand m to (... Bq 1)


                updated_max_exp = torch.exp(m_prev - m) # (batch, Bq)
                # Compute l: Running exp sum per row (batch, Bq) (not accurate bc max is changing)
                l = updated_max_exp * l + torch.sum(p_partial, dim=-1)


                # Compute Oi(j): jth iteration of the ith tile of o (at this point it's the numerator being updated by the new max across the seq)
                o_tile = einsum(torch.diag(updated_max_exp), o_tile, "... Bq j, ... j d -> ... Bq d") + einsum(p_partial, v_tile, "... Bq Bk, Bk d -> ... Bq d")
            
            # Compute Oi using the latest iteration and normalizing
            o_tile = einsum(torch.inverse(torch.diag(l)), o_tile, "... Bq i, ... i d -> ... Bq d")
            # Write to output
            output[:, i*Bq:(i+1)*Bq, :] = o_tile

            # Compute logsumexp
            logsumexp[:, i*Bq:(i+1)*Bq] = m + torch.log(l)
        
        return output, logsumexp








    @staticmethod
    def backward(ctx):
        raise NotImplementedError