import torch
from transformers import BartConfig

def get_dnabart_config():
    """
    Returns a customized BartConfig for DNABART.

    Returns:
        BartConfig: Customized configuration object.
    """    
    config = BartConfig(
        vocab_size=4096,            # Adjust based on your tokenizer
        d_model=768,                # Hidden size of the encoder and decoder
        encoder_layers=6,           # Number of encoder layers
        decoder_layers=6,           # Number of decoder layers
        encoder_attention_heads=12, # Number of attention heads in the encoder
        decoder_attention_heads=12, # Number of attention heads in the decoder
        encoder_ffn_dim=3072,       # Feed-forward network dimension in encoder
        decoder_ffn_dim=3072,       # Feed-forward network dimension in decoder
        max_position_embeddings=512, # Maximum position embeddings
        dropout=0.1,                # Dropout rate
        attention_dropout=0.1,      # Dropout rate for attention layers
        activation_function="gelu", # Activation function
        pad_token_id=0,
        unk_token_id=1,
        sep_token_id=2,
        mask_token_id=3,
        bos_token_id=4,
        eos_token_id=5
    )
    return config


def get_device():
    """
    Returns the appropriate device (CUDA if available, otherwise CPU).

    Returns:
        torch.device: The device object.
    """
    if torch.cuda.is_available():
        print("Using cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")