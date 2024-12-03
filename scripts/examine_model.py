import torch
from omegaconf import OmegaConf
import os
from apps.main.transformer import LMTransformerArgs, LMTransformer

def examine_model():
    # Create complete model args
    model_args = LMTransformerArgs(
        dim=1024,
        n_layers=8,
        n_heads=8,
        vocab_size=64772,  # From config
        init_base_std=0.02,  # Common value for transformer models
        init_std_factor="disabled",  # Default value
        max_seqlen=16384,  # From config's seq_len
        norm_eps=1e-5,  # Default value
        rope_theta=10000.0,  # Default value
        weight_tying=False,  # Default value
        sliding_window=None  # Default value
    )
    
    # Build model
    model = LMTransformer(model_args)
    
    # Print embedding layer info
    print("\nEmbedding Layer:")
    print(f"Shape: {model.tok_embeddings.weight.shape}")
    print(f"Number of embeddings: {model.tok_embeddings.num_embeddings}")
    print(f"Embedding dimension: {model.tok_embeddings.embedding_dim}")
    
    # Print lm_head info
    print("\nLM Head Layer:")
    print(f"Shape: {model.output.weight.shape}")
    print(f"Input features: {model.output.in_features}")
    print(f"Output features: {model.output.out_features}")
    
    # Print config values
    print("\nConfig Values:")
    print(f"model.vocab_size: {model_args.vocab_size}")
    print(f"model.dim: {model_args.dim}")
    
    # Print max token ID that can be handled
    max_vocab = max(model.tok_embeddings.num_embeddings, model.output.out_features)
    print(f"\nMaximum token ID that can be handled: {max_vocab - 1}")

if __name__ == "__main__":
    examine_model() 