from networks.UNet import UNet
from networks.ViT import ViT
from model_diff import VDM_Tools, DDPM_Tools


def create_model_diffusion(device, **kwargs):
    model = create_model(
        model_name=kwargs.get("model_name"),
        n_blocks=kwargs.get("n_blocks"),
        n_channels=kwargs.get("n_channels"),
        n_heads=kwargs.get("n_heads"),
        embed_dim=kwargs.get("embed_dim"),
        device=device
    )

    diffusion = create_diffusion(
        noise_steps=kwargs.get("noise_steps"),
        device=device,
        conditioned_prior=kwargs.get("conditioned_prior"),
        noise_schedule=kwargs.get("noise_schedule"),
        normalization_factors=kwargs.get("normalization_factors"),
        loss=kwargs.get("loss")
    )

    return model, diffusion

def create_model(
        model_name,
        n_blocks,
        n_channels,
        n_heads,
        embed_dim,
        device
):
    if model_name == "UNet":
        return UNet(
            input_channels=2,
            output_channels=1,
            n_blocks=n_blocks,
            n_channels=n_channels
        ).to(device)

    elif model_name == "ViT":
        return ViT(
            depth=n_blocks,
            num_heads=n_heads,
            embed_dim=embed_dim
        ).to(device)

    else:
        raise ValueError("Model Type not supported")

def create_diffusion(
        noise_steps,
        device,
        conditioned_prior,
        noise_schedule,
        normalization_factors,
        loss
):
    if loss == "l2":
        return DDPM_Tools(
            noise_steps=noise_steps,
            conditioned_prior=conditioned_prior,
            noise_schedule=noise_schedule,
            loss=loss,
            device=device
        )
    elif loss == "vlb":
        return VDM_Tools(
            noise_steps=noise_steps,
            conditioned_prior=conditioned_prior,
            noise_schedule=noise_schedule,
            loss=loss,
            device=device
        )
    else:
        raise ValueError("Loss Type not supported")

