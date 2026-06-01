
import math

import torch
import triton
import triton.language as tl



@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.contstexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)


    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_QUERIES, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_QUERIES, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    # Initialize a buffer to write to

    acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    m = tl.full((Q_TILE_SIZE, ), -float("inf"), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE, ), dtype=tl.float32)

    Q_tile = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero") # (Q_tile_size, D)

    # Form bq_index
    if is_causal:
        Bq_index = tl.arange(query_tile_index * Q_TILE_SIZE, (1+query_tile_index) * Q_TILE_SIZE)

    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)): # (Go through each of the sets of keys)
        # Load Q K V tile
        K_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_tile_size, D)
        V_tile = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero") # (K_tile_size, D)

        # Compute S: (batch, Bq, Bd) Representing the scores for this tile
        # This is EXACT since we're loading the whole row and column (K and Q) per tile

        

        # Perform masking
        if is_causal:
            # Form mask

            ## Form Bk index vector
            Bk_index = tl.arange(i * K_TILE_SIZE, (1+i) * K_TILE_SIZE)

            ## Compare to create Bq * Bk mask
            mask = Bq_index[:, None] < Bk_index[None, :]

            # Mask out S
            S = tl.where(mask, -1e6, tl.dot(Q_tile, tl.trans(K_tile)) * scale)
        
        else:
            S = tl.dot(Q_tile, tl.trans(K_tile)) * scale



        m_prev = m
        m = tl.maximum(m, tl.max(S, dim=-1)) # (Q_TILE_SIZE,)
        P_partial = tl.exp(S - m[:, None]) # (Q_TILE_SIZE, K_TILE_SIZE)

        alpha = tl.exp(m_prev - m) # (Q_TILE_SIZE,)
        l = alpha * l + tl.sum(P_partial, dim=-1) # (Q_TILE_SIZE,)

        acc = alpha[:, None] * acc
        acc = tl.dot(P_partial.to(V_tile.dtype), V_tile, acc=acc).to() # (Q_TILE_SIZE, D)

        # Advance pointers for K and V
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # Compute O tile
    o = (acc / l[:, None]).to(Q_block_ptr.type.element_ty)

    # Compute L tile
    L = m + tl.log(l)
    
    # Store result in O and L pointers
    tl.store(O_block_ptr, o, boundary_check=(0, 1))
    tl.store(L_block_ptr, L, boundary_check=(0,))




class FlashAttention2Triton(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal=False):


        BATCH_SIZE, N_QUERIES, D = q.shape
        N_KEYS = k.shape[-2]

        scale = 1 / math.sqrt(D)

        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16

        # Initialize empty O and L
        o = torch.empty(q.shape, device=q.device) # (..., N_QUERIES, D)
        L = torch.empty((BATCH_SIZE, Q_TILE_SIZE), device=q.device)

        # Call kernel
        flash_fwd_kernel[(math.ceil(N_QUERIES / Q_TILE_SIZE), BATCH_SIZE)](
            q, k, v,
            o, L,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            L.stride(0), L.stride(2),
            N_QUERIES, N_KEYS,
            scale,
            D,
            Q_TILE_SIZE,
            K_TILE_SIZE,
            is_causal
        )


        # Save parameters for backward pass
        ctx.save_for_backward(L, q, k, v, o)
        ctx.is_causal = is_causal

        return o
    


    @staticmethod
    def backward(ctx):
        # We need to calculate dQ dK and dV, we receive dO upstream

        raise NotImplementedError