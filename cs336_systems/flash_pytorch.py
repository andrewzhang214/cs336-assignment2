
import math

from einops import rearrange, einsum, reduce
import torch


def backward(L, q, k, v, o, dO):

    # q (batch, Nq, d)
    # k (batch, Nk, d)
    # v (batch, Nk, d)

    # L (batch, Nq)
    # dO (batch, Nq, d)

    # Return dQ, dK, dV

    batch, Nq, d = q.shape

    scale = 1 / math.sqrt(d)

    S = einsum(q, k, "... Nq d, ... Nk d -> batch Nq Nk") * scale
    





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
        o = torch.empty(batch, seq_len, d, device=device, dtype=torch.float32)
        L = torch.empty(batch, seq_len, device=device, dtype=torch.float32)

        for i in range(Tq): # Iterates over sets of query vectors
            # Load Qi from VRAM (batch, Bq, d)
            q_tile = q[:, i*Bq:(i+1)*Bq, :]

            # Initialize Oi_0 (batch, Bq, d) 
            o_tile = torch.zeros(batch, Bq, d, device=device, dtype=torch.float32)

            # Initialize li (batch, Bq)  Running sum of exponentials per row (Changes as we go across seq)
            l = torch.zeros(batch, Bq, device=device, dtype=torch.float32)

            # Initialize mi (batch, Bq)  Running max per row (Changes as we go across seq)
            m = torch.full((batch, Bq), -float("inf"), device=device, dtype=torch.float32)

            for j in range(Tk): # Iterates over sets of key and value vectors and now iteratively creates O horizontally across the seq (O shape stays the same and essentially is the running average over the value vectors as we iterate over them)
                # Load Kj and Vj (batch, Bk, d) 
                k_tile = k[:, j*Bk:(j+1)*Bk, :]
                v_tile = v[:, j*Bk:(j+1)*Bk, :]

                # Compute S: (batch, Bq, Bd) Representing the scores for this tile
                # This is EXACT since we're loading the whole row and column (K and Q) per tile
                s = einsum(q_tile, k_tile, "... Bq d, ... Bk d -> ... Bq Bk") / scale


                # Compute m: Running max per row (batch, Bq, 1)
                s_row_max = reduce(s, "... Bq Bk -> ... Bq", "max")
                m_prev = m.clone()
                m = rearrange(torch.maximum(s_row_max, m), "... Bq -> ... Bq 1")
                m_prev = rearrange(m_prev, "... Bq -> ... Bq 1")

                # Compute partial P (batch, Bq, Bk)
                # This is NOT EXACT since the running max is changing (also it's not normalized yet)

                p_partial = torch.exp(s - m)

                m_prev = rearrange(m_prev, "... Bq 1 -> ... Bq")
                m = rearrange(m, "... Bq 1 -> ... Bq")

                updated_max_exp = torch.exp(m_prev - m) # (batch, Bq)
                # Compute l: Running exp sum per row (batch, Bq) (not accurate bc max is changing)
                l = updated_max_exp * l + torch.sum(p_partial, dim=-1)


                # Compute Oi(j): jth iteration of the ith tile of o (at this point it's the numerator being updated by the new max across the seq)
                o_tile = einsum(torch.diag_embed(updated_max_exp), o_tile, "... Bq j, ... j d -> ... Bq d") + einsum(p_partial, v_tile, "... Bq Bk, ... Bk d -> ... Bq d")
            
            # Compute Oi using the latest iteration and normalizing
            o_tile = einsum(torch.inverse(torch.diag_embed(l)), o_tile, "... Bq i, ... i d -> ... Bq d")
            # Write to output
            o[:, i*Bq:(i+1)*Bq, :] = o_tile

            # Compute logsumexp
            L[:, i*Bq:(i+1)*Bq] = m + torch.log(l)

        # Save ctx
        ctx.save_for_backward(L, q, k, v, o)

        # Memory complexity
        # L: O(batch * seq_len)
        # q, k, v: O(batch * seq_len * d)
        # o: O(batch * seq_len * d)
        
        return o


    @staticmethod
    def backward(ctx):
        # We need to calculate dQ dK and dV, we receive dO upstream

        raise NotImplementedError