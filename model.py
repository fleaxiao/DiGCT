import argparse
import torch
import os
import yaml
import json
import datetime
import numpy as np

from torch import optim

from model_train import ModelTrainer
from model_dataset import get_data
from model_test import sample_model_output, calculate_metrics, sample_save_metrics
from model_utils import save_line_chart, set_seed, load_images
from model_model import create_model_diffusion


def main(args):

    # Load arguments
    DEFAULT_SEED = args.default_seed 
    INPUT_PATH = args.input_path
    OUTPUT_PATH = args.output_path
    DATA_PATH = args.data_path
    TARGET_FOLDER = args.target_folder
    CONDITION_FOLDER = args.condition_folder

    TRAINING = args.training
    GENERATE_SAMPLE = args.generate_sample
    PHYSICS_INFORMED = args.physics_informed
    POLAR_CNN = args.polar_cnn
    CONDITIONED_PRIOR = args.conditioned_prior
    THRESHOLD_TRAINING = args.threshold_training
    CLIP_GRAD = args.clip_grad
    EMA = args.ema
    NOISE_SCHEDULE = args.noise_schedule
    LOSS = args.loss

    RESOLUTION = args.resolution
    TEST_SPLIT = args.test_split
    VALIDATION_SPLIT = args.validation_split
    EPOCHS = args.epochs
    NOISE_STEPS = args.noise_steps
    BATCH_SIZE = args.batch_size
    INIT_LR = args.init_lr
    WEIGHT_DECAY = args.weight_decay
    THRESHOLD = args.threshold
    SAMPLE_NUMBER = args.sample_number
    SAMPLE_EPOCH = args.sample_epoch
    EMA_DECAY = args.ema_decay
    
    TESTING = args.testing
    CALCULATE_METRICS = args.calculate_metrics
    SAMPLE_METRICS = args.sample_metrics
    TEST_PATH = args.test_path
    SAMPLE_MODEL = args.sample_model
    NR_SAMPLES = args.nr_samples

    MODEL_NAME = args.model_name
    BLOCKS = args.n_blocks
    CHANNELS = args.n_channels

    DATASET_PATH = os.path.join(INPUT_PATH, DATA_PATH) 
    TARGET_DATASET_PATH = os.path.join(DATASET_PATH, TARGET_FOLDER)
    CONDITION_DATASET_PATH = os.path.join(DATASET_PATH, CONDITION_FOLDER)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parameters = {
        "model_name": MODEL_NAME,
        "threshold_training": THRESHOLD_TRAINING,
        "test_split": TEST_SPLIT,
        "validation_split": VALIDATION_SPLIT,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "init_lr": INIT_LR,
        "weight_decay": WEIGHT_DECAY,
        "threshold": THRESHOLD,
        "sample_number": SAMPLE_NUMBER,
        "sample_epoch": SAMPLE_EPOCH,
        "ema_decay": EMA_DECAY,
        "noise_steps": NOISE_STEPS,
        "n_blocks": BLOCKS,
        "n_channels": CHANNELS,
        "resolution": RESOLUTION,
        "clip_grad": CLIP_GRAD,
        "physics_informed": PHYSICS_INFORMED,
        "polar_cnn": POLAR_CNN,
        "conditioned_prior": CONDITIONED_PRIOR,
        "noise_schedule": NOISE_SCHEDULE,
        "loss": LOSS
    }

    # Training Configuration
    if TRAINING:
        run_inst = 0
        RUN_NAME = f"{MODEL_NAME}_bs({BATCH_SIZE})_noisesteps({NOISE_STEPS})_block({BLOCKS})_loss({LOSS})_{run_inst}"
        while os.path.exists(os.path.join(OUTPUT_PATH, RUN_NAME)):
            run_inst += 1
            RUN_NAME = f"{MODEL_NAME}_bs({BATCH_SIZE})_noisesteps({NOISE_STEPS})_block({BLOCKS})_loss({LOSS})_{run_inst}"

        ## paths
        RESULT_PATH = os.path.join(OUTPUT_PATH, RUN_NAME)
        MODEL_PATH = os.path.join(RESULT_PATH, "model")
        TRAIN_PATH = os.path.join(RESULT_PATH, "train")
        SAMPLE_PATH = os.path.join(RESULT_PATH, "sample")
        TARGET_PATH = os.path.join(SAMPLE_PATH, "target")
        GENERATION_PATH = os.path.join(SAMPLE_PATH, "generation")
        CONDITION_PATH = os.path.join(SAMPLE_PATH, "condition")

        for path in [MODEL_PATH, TRAIN_PATH, SAMPLE_PATH, TARGET_PATH, GENERATION_PATH, CONDITION_PATH]:
            os.makedirs(path, exist_ok=True)

        with open(os.path.join(RESULT_PATH, 'parameters.json'), "w") as f:
            json.dump(parameters, f, indent=4)
        
        print(f"Experiment Name: {RUN_NAME}")
        start_time = datetime.datetime.now()
        print(f"Experiment Starting Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Experiment Device: {DEVICE}")

        ## seed
        set_seed(seed=DEFAULT_SEED)
        print("Experiment Seed: set\n")

        ## dataloader
        train_dataloader, val_dataloader, test_dataloader, _, _, _ = get_data(dataset_path=DATASET_PATH, target_dataset_path=TARGET_DATASET_PATH, condition_dataset_path=CONDITION_DATASET_PATH, result_path=RESULT_PATH, **parameters)
        
        ## model
        model, diffusion = create_model_diffusion(DEVICE, **parameters)
        
        ## trainer
        optimizer = optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
        trainer = ModelTrainer(model=model,
                            device=DEVICE,
                            optimizer=optimizer,
                            result_path=RESULT_PATH,
                            train_path=TRAIN_PATH,
                            model_path=MODEL_PATH,
                            train_dataloader=train_dataloader,
                            val_dataloader=val_dataloader,
                            test_dataloader=test_dataloader, 
                            diffusion=diffusion,
                            ema=EMA,
                            **parameters)
        trainer.train()

        ## save results
        torch.save(trainer.best_model_checkpoint, os.path.join(MODEL_PATH, "best_model.pth"))
        if trainer.ema == True and trainer.ema_model is not None:
            torch.save(trainer.ema_model.state_dict(), os.path.join(MODEL_PATH, "ema_model.pth"))

        if CONDITIONED_PRIOR == True:
            torch.save(trainer.diffusion.prior_mean, os.path.join(OUTPUT_PATH, "prior_mean.pth"))
            torch.save(trainer.diffusion.prior_variance, os.path.join(OUTPUT_PATH, "prior_variance.pth"))

        max_ssim = max(trainer.ssim_values)
        min_mae = min(trainer.mae_values)
        print(f"\nMax SSIM: {max_ssim:.4f} (epoch: {SAMPLE_EPOCH * (np.argmax(trainer.ssim_values) + 1)})")
        print(f"Min MAE: {min_mae:.4f} (epoch: {SAMPLE_EPOCH * (np.argmin(trainer.mae_values) + 1)})")
        print(f"Min Val Loss: {trainer.best_val_loss:.4f} -> Best Model (epoch: {trainer.best_model_epoch + 1})")

        end_time = datetime.datetime.now()
        print(f"\nExperiment Name: {RUN_NAME}")
        print(f"Experiment Ending Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Experiment Duration: {(end_time - start_time).total_seconds() / 60:.2f} minutes")

        results = {
            "max SSIM": float(max_ssim),
            "max SSIM epoch": int(SAMPLE_EPOCH * (np.argmax(trainer.ssim_values) + 1)),
            "min MAE": float(min_mae),
            "min MAE epoch": int(SAMPLE_EPOCH * (np.argmin(trainer.mae_values) + 1)),
            "best model epoch": int(trainer.best_model_epoch + 1)
        }
        with open(os.path.join(RESULT_PATH, 'results.json'), "w") as f:
            json.dump(results, f, indent=4)

        np.savez(os.path.join(RESULT_PATH, "losses_train.npz"), losses=trainer.train_losses)
        np.savez(os.path.join(RESULT_PATH, "losses_val.npz"), losses=trainer.val_losses)
        np.savez(os.path.join(RESULT_PATH, "results_ssim.npz"), ssim=trainer.ssim_values)
        np.savez(os.path.join(RESULT_PATH, "results_mae.npz"), mae=trainer.mae_values)

        save_line_chart(data = trainer.ssim_values, sample_epoch = SAMPLE_EPOCH, title = 'SSIM', path = os.path.join(RESULT_PATH, "results_ssim.png"))
        save_line_chart(data = trainer.mae_values, sample_epoch = SAMPLE_EPOCH, title = 'MAE', path = os.path.join(RESULT_PATH, "results_mae.png"))

        if GENERATE_SAMPLE == True:
            if EMA == True:
                model = trainer.ema_model
            else:
                model.load_state_dict(trainer.best_model_checkpoint)
            sample_save_metrics(model=model, device=DEVICE, sampler=trainer.diffusion, length=(len(test_dataloader) - 1) * BATCH_SIZE,                    test_dataloader=test_dataloader, sample_path=SAMPLE_PATH, **parameters)

    if TESTING:
        PARAMETER_PATH = os.path.join(TEST_PATH, 'parameters.json')
        MODEL_PATH = os.path.join(os.path.join(TEST_PATH, "models"), SAMPLE_MODEL)
        SAMPLE_PATH = os.path.join(TEST_PATH, "sample")
        TARGET_PATH = os.path.join(SAMPLE_PATH, "target")
        GENERATION_PATH = os.path.join(SAMPLE_PATH, "generation")
        CONDITION_PATH = os.path.join(SAMPLE_PATH, "condition")
        TEST_DATASET_PATH = os.path.join(TEST_PATH, "test_indices.pth")

        dataloader, _, _, _, _, _ = get_data(target_dataset_path=TARGET_DATASET_PATH, condition_dataset_path=CONDITION_DATASET_PATH, split=False, **parameters)

        if CALCULATE_METRICS == True:
            condition_images = load_images(CONDITION_PATH)
            target_images = load_images(TARGET_PATH)
            sample_images = load_images(SAMPLE_PATH)
            ssim_values, psnr_values, mse_mean_values, mse_max_values, mae_values = calculate_metrics(target_images[0:1], sample_images[0:1])
            print(f"SSIM: {np.mean(ssim_values)}, PSNR: {np.mean(psnr_values)}, MAE: {np.mean(mae_values)}, MSE Mean: {np.mean(mse_mean_values)}, MSE Max: {np.mean(mse_max_values)}")

        if SAMPLE_METRICS == True:
            with open(PARAMETER_PATH, "r") as f:
                parameters = json.load(f)

            print(f"Sampling model: {MODEL_PATH}")
            set_seed(seed=DEFAULT_SEED)
            model, sampler = create_model_diffusion(DEVICE, **parameters)
            sampler.load_prior_mean_variance(os.path.join(TEST_PATH, "models"))
            model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
            sample_save_metrics(model=model, device=DEVICE,sampler=sampler, length=NR_SAMPLES, test_dataloader=dataloader, sample_path=SAMPLE_PATH, **parameters)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='DiGCT Model')
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/config_model.yml')

    # Seed
    p.add_argument('--default_seed', action='store', type=int, default=42)

    # Paths
    p.add_argument('--input_path', action='store', type=str, default='dataset')
    p.add_argument('--output_path', action='store', type=str, default='results')
    p.add_argument('--data_path', action='store', type=str, default='IGCT')
    p.add_argument('--target_folder', action='store', type=str, default='target')
    p.add_argument('--condition_folder', action='store', type=str, default='condition')

    # Training settings
    p.add_argument('--training', action='store_true', default=False)
    p.add_argument('--generate_sample', action='store_true', default=False)

    p.add_argument('--physics_informed', action='store_true', default=False)
    p.add_argument('--polar_cnn', action='store_true', default=False)
    p.add_argument('--conditioned_prior', action='store_true', default=False)
    p.add_argument('--threshold_training', action='store_true', default=False)
    p.add_argument('--clip_grad', action='store_true', default=False)
    p.add_argument('--ema', action='store_true', default=False)

    p.add_argument('--resolution', action='store', type=int, default=64)
    p.add_argument('--test_split', action='store', type=float, default=0.1)
    p.add_argument('--validation_split', action='store', type=float, default=0.1)
    p.add_argument('--noise_schedule', action='store', type=str, default='fixed_linear')
    p.add_argument('--loss', action='store', type=str, default='l2')

    # Training parameters
    p.add_argument('--epochs', action='store', type=int, default=100)
    p.add_argument('--noise_steps', action='store', type=int, default=1000)
    p.add_argument('--batch_size', action='store', type=int, default=16)
    p.add_argument('--init_lr', action='store', type=float, default=2e-4)
    p.add_argument('--weight_decay', action='store', type=float, default=0.0)
    p.add_argument('--threshold', action='store', type=float, default=0.01)
    p.add_argument('--sample_number', action='store', type=int, default=2)
    p.add_argument('--sample_epoch', action='store', type=int, default=5)
    p.add_argument('--ema_decay', action='store', type=float, default=0.995)

    # Testing parameters
    p.add_argument('--testing', action='store_true', default=False)
    p.add_argument('--calculate_metrics', action='store_true', default=False)
    p.add_argument('--sample_metrics', action='store_true', default=False)
    p.add_argument('--test_path', action='store', type=str, default='results')
    p.add_argument('--nr_samples', action='store', type=int, default=100)

    # Model parameters
    p.add_argument('--model_name', action='store', type=str, default='UNet')
    p.add_argument('--n_blocks', action='store', type=int, default=2)
    p.add_argument('--n_channels', action='store', type=int, default=64)

    # Load config file
    args = p.parse_args()
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
                arg_dict[key] = value
        args.config = args.config.name
    else:
        config_dict = {}

    main(args=args)