import torch

def get_encoder_output_size(encoder, dims):
    x = torch.randn((1,)+dims)
    with torch.no_grad():
        out = encoder(x)
    return list(out.size())[1:]