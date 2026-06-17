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
        conditioned_prior=kwargs.get("conditioned_prior"),
        noise_schedule=kwargs.get("noise_schedule"),
        loss=kwargs.get("loss"),
        physics_constraint=kwargs.get("physics_constraint"),
        lambda_physics=kwargs.get("lambda_physics"),
        device=device,
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
            input_channels=1,
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
        conditioned_prior,
        noise_schedule,
        loss,
        physics_constraint,
        lambda_physics,
        device,
):
    if loss == "l2":
        return DDPM_Tools(
            noise_steps=noise_steps,
            conditioned_prior=conditioned_prior,
            noise_schedule=noise_schedule,
            loss=loss,
            physics_constraint=physics_constraint,
            lambda_physics=lambda_physics,
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

