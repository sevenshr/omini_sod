import lightning as L
from diffusers.pipelines import FluxPipeline
import torch
import wandb
import os
import yaml
from peft import LoraConfig, get_peft_model_state_dict
from torch.utils.data import DataLoader
import time

from typing import List

import prodigyopt

from ..pipeline.flux_omini import transformer_forward, encode_images
import utils
import torch.nn.functional as F
from ..pipeline.flux_omini import Condition, convert_to_condition, generate
from torchmetrics import Metric

def get_rank():
    try:
        rank = int(os.environ.get("LOCAL_RANK"))
    except:
        rank = 0
    return rank


def get_config():
    config_path = os.environ.get("OMINI_CONFIG")
    assert config_path is not None, "Please set the OMINI_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def init_wandb(wandb_config, run_name):
    import wandb

    try:
        os.environ["WANDB_API_KEY"] = "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"
        wandb.login(key="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")
        assert os.environ.get("WANDB_API_KEY") is not None
        wandb.init(
            project=wandb_config["project"],
            name=run_name,
            config={},
        )
    except Exception as e:
        print("Failed to initialize WanDB:", e)


class OminiModel(L.LightningModule):
    def __init__(
        self,
        flux_pipe_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        adapter_names: List[str] = [None, None, "default"],
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config

        # Load the Flux pipeline
        self.flux_pipe: FluxPipeline = FluxPipeline.from_pretrained(
            flux_pipe_id, torch_dtype=dtype
        ).to(device)
        self.transformer = self.flux_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()

        # Freeze the Flux pipeline
        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()
        self.adapter_names = adapter_names
        self.adapter_set = set([each for each in adapter_names if each is not None])

        # Initialize LoRA layers
        self.lora_layers = self.init_lora(lora_path, lora_config)

        self.to(device).to(dtype)

        self.val_metric1 = utils.DDPAverager()
        self.val_metric2 = utils.DDPAverager()
        self.val_metric3 = utils.DDPAverager()
        self.val_metric4 = utils.DDPAverager()
        self.val_metric5 = utils.DDPAverager()
        self.metric_fn = utils.calc_cod_single


    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config
        if lora_path:
            # TODO: Implement this
            raise NotImplementedError
        else:
            for adapter_name in self.adapter_set:
                self.transformer.add_adapter(
                    LoraConfig(**lora_config), adapter_name=adapter_name
                )
            # TODO: Check if this is correct (p.requires_grad)
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        return list(lora_layers)

    def save_lora(self, path: str):
        for adapter_name in self.adapter_set:
            FluxPipeline.save_lora_weights(
                save_directory=path,
                weight_name=f"{adapter_name}.safetensors",
                transformer_lora_layers=get_peft_model_state_dict(
                    self.transformer, adapter_name=adapter_name
                ),
                safe_serialization=True,
            )

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = self.lora_layers

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError("Optimizer not implemented.")
        return optimizer

    def training_step(self, batch, batch_idx):
        imgs, prompts = batch["image"], batch["description"]
        image_latent_mask = batch.get("image_latent_mask", None)

        # Get the conditions and position deltas from the batch
        conditions, position_deltas, position_scales, latent_masks = [], [], [], []
        for i in range(1000):
            if f"condition_{i}" not in batch:
                break
            conditions.append(batch[f"condition_{i}"])
            position_deltas.append(batch.get(f"position_delta_{i}", [[0, 0]]))
            position_scales.append(batch.get(f"position_scale_{i}", [1.0])[0])
            latent_masks.append(batch.get(f"condition_latent_mask_{i}", None))

        # Prepare inputs
        with torch.no_grad():
            # Prepare image input
            x_0, img_ids = encode_images(self.flux_pipe, imgs)

            # Prepare text input
            (
                prompt_embeds,
                pooled_prompt_embeds,
                text_ids,
            ) = self.flux_pipe.encode_prompt(
                prompt=prompts,
                prompt_2=None,
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                device=self.flux_pipe.device,
                num_images_per_prompt=1,
                max_sequence_length=self.model_config.get("max_sequence_length", 512),
                lora_scale=None,
            )

            # Prepare t and x_t
            # t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            t = torch.ones((imgs.shape[0],), device=self.device)
            x_1 = torch.randn_like(x_0).to(self.device)
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)
            if image_latent_mask is not None:
                x_0 = x_0[:, image_latent_mask[0]]
                x_1 = x_1[:, image_latent_mask[0]]
                x_t = x_t[:, image_latent_mask[0]]
                img_ids = img_ids[image_latent_mask[0]]

            # Prepare conditions
            condition_latents, condition_ids = [], []
            for cond, p_delta, p_scale, latent_mask in zip(
                conditions, position_deltas, position_scales, latent_masks
            ):
                # Prepare conditions
                c_latents, c_ids = encode_images(self.flux_pipe, cond)
                # Scale the position (see OminiConrtol2)
                if p_scale != 1.0:
                    scale_bias = (p_scale - 1.0) / 2
                    c_ids[:, 1:] *= p_scale
                    c_ids[:, 1:] += scale_bias
                # Add position delta (see OminiControl)
                c_ids[:, 1] += p_delta[0][0]
                c_ids[:, 2] += p_delta[0][1]
                if len(p_delta) > 1:
                    print("Warning: only the first position delta is used.")
                # Append to the list
                if latent_mask is not None:
                    c_latents, c_ids = c_latents[latent_mask], c_ids[latent_mask[0]]
                condition_latents.append(c_latents)
                condition_ids.append(c_ids)

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )

        branch_n = 2 + len(conditions)
        group_mask = torch.ones([branch_n, branch_n], dtype=torch.bool).to(self.device)
        # Disable the attention cross different condition branches
        group_mask[2:, 2:] = torch.diag(torch.tensor([1] * len(conditions)))
        # Disable the attention from condition branches to image branch and text branch
        if self.model_config.get("independent_condition", False):
            group_mask[2:, :2] = False

        # Forward pass
        transformer_out = transformer_forward(
            self.transformer,
            image_features=[x_t, *(condition_latents)],
            text_features=[prompt_embeds],
            img_ids=[img_ids, *(condition_ids)],
            txt_ids=[text_ids],
            # There are three timesteps for the three branches
            # (text, image, and the condition)
            timesteps=[t, t] + [torch.zeros_like(t)] * len(conditions),
            # Same as above
            pooled_projections=[pooled_prompt_embeds] * branch_n,
            guidances=[guidance] * branch_n,
            # The LoRA adapter names of each branch
            adapters=self.adapter_names,
            return_dict=False,
            group_mask=group_mask,
        )
        pred = transformer_out[0]

        # Compute loss
        step_loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
        self.last_t = t.mean().item()

        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    def generate_a_sample(self):
        raise NotImplementedError("Generate a sample not implemented.")


    def test_step(self, batch, batch_idx):
        imgs, prompts = batch["image"], batch["description"]
        image_latent_mask = batch.get("image_latent_mask", None)
        conditions_use, conditions, position_deltas, position_scales, latent_masks = [], [], [], [], []
        adapter = self.adapter_names[2]
        for i in range(1000):
            if f"condition_{i}" not in batch:
                break
            conditions.append(batch[f"condition_{i}"])
            position_deltas.append(batch.get(f"position_delta_{i}", [0, 0]))
            position_scales.append(batch.get(f"position_scale_{i}", [1.0])[0])
            latent_masks.append(batch.get(f"condition_latent_mask_{i}", None))
            conditions_use.append(Condition(conditions[-1][0], adapter, position_deltas[-1][0], position_scales[-1]))
        generator = torch.Generator(device=self.device)
        generator.manual_seed(42)
        pre_res = generate(
            self.flux_pipe,
            prompt=prompts,
            conditions=[conditions_use[0]],
            height=imgs.shape[-1],
            width=imgs.shape[-2],
            generator=generator,
            model_config=self.model_config,
            kv_cache=self.model_config.get("independent_condition", False),
            output_type = "pt",
            num_inference_steps = 1,
        )
        res = pre_res.images[0].mean(dim=0).unsqueeze(0).unsqueeze(0)
        res = F.interpolate(res, size=imgs.shape[-2:], mode='bilinear', align_corners=False)
        res = res.sigmoid().cpu().float().numpy().squeeze()
        imgs = imgs.cpu().float().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        result1, result2, result3, result4, result5 = self.metric_fn(res, imgs)
        
        self.val_metric1.update(torch.tensor(result1.item()).to(self.device), 1)
        self.val_metric2.update(torch.tensor(result2.item()).to(self.device), 1)
        self.val_metric3.update(torch.tensor(result3.item()).to(self.device), 1)
        self.val_metric4.update(torch.tensor(result4.item()).to(self.device), 1)
        self.val_metric5.update(torch.tensor(result5.item()).to(self.device), 1)

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)


class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}, test_function=None):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )

        self.total_steps = 0
        self.test_function = test_function

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        # Print training progress every n steps
        if self.use_wandb:
            report_dict = {
                "steps": batch_idx,
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t
            wandb.log(report_dict)

        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        # Save LoRA weights at specified intervals
        if self.total_steps % self.save_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights"
            )
            pl_module.save_lora(
                f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            )

        # Generate and save a sample image at specified intervals
        if self.total_steps % self.sample_interval == 0 and self.test_function:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating a sample"
            )
            # pl_module.eval()
            # self.test_function(
            #     pl_module,
            #     f"{self.save_path}/{self.run_name}/output",
            #     f"lora_{self.total_steps}",
            # )
            # pl_module.train()
    def on_validation_epoch_end(self, trainer, pl_module):
        m1 = pl_module.val_metric1.compute().cpu().numpy()
        m2 = pl_module.val_metric2.compute().cpu().numpy()
        m3 = pl_module.val_metric3.compute().cpu().numpy()
        m4 = pl_module.val_metric4.compute().cpu().numpy()
        m5 = pl_module.val_metric5.compute().cpu().numpy()

        print('metric1: {:.4f}'.format(m1))
        print('metric2: {:.4f}'.format(m2))
        print('metric3: {:.4f}'.format(m3))
        print('metric4: {:.4f}'.format(m4))
        print('metric5: {:.4f}'.format(m5))
        save_path = f"{self.save_path}/{self.run_name}/output"
        os.makedirs(save_path, exist_ok=True)
        file_name = f"lora_{self.total_steps}"
        log_path = os.path.join(save_path, 'val_metrics.txt')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"{file_name} metric1: {m1:.6f}, metric2: {m2:.6f}, metric3: {m3:.6f}, metric4: {m4:.6f}, metric5: {m5:.6f}\n")

        # Log to WandB
        if self.use_wandb:
            wandb.log({
                "val_metric1": m1,
                "val_metric2": m2,
                "val_metric3": m3,
                "val_metric4": m4,
                "val_metric5": m5,
            })

        pl_module.val_metric1.reset()
        pl_module.val_metric2.reset()
        pl_module.val_metric3.reset()
        pl_module.val_metric4.reset()
        pl_module.val_metric5.reset()

    def on_test_epoch_end(self, trainer, pl_module):
        m1 = pl_module.val_metric1.compute().cpu().numpy()
        m2 = pl_module.val_metric2.compute().cpu().numpy()
        m3 = pl_module.val_metric3.compute().cpu().numpy()
        m4 = pl_module.val_metric4.compute().cpu().numpy()
        m5 = pl_module.val_metric5.compute().cpu().numpy()

        print('metric1: {:.4f}'.format(m1))
        print('metric2: {:.4f}'.format(m2))
        print('metric3: {:.4f}'.format(m3))
        print('metric4: {:.4f}'.format(m4))
        print('metric5: {:.4f}'.format(m5))
        save_path = f"{self.save_path}/{self.run_name}/output"
        os.makedirs(save_path, exist_ok=True)
        file_name = f"lora_{self.total_steps}"
        log_path = os.path.join(save_path, 'val_metrics.txt')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"{file_name} metric1: {m1:.6f}, metric2: {m2:.6f}, metric3: {m3:.6f}, metric4: {m4:.6f}, metric5: {m5:.6f}\n")

        # Log to WandB
        if self.use_wandb:
            wandb.log({
                "val_metric1": m1,
                "val_metric2": m2,
                "val_metric3": m3,
                "val_metric4": m4,
                "val_metric5": m5,
            })

        pl_module.val_metric1.reset()
        pl_module.val_metric2.reset()
        pl_module.val_metric3.reset()
        pl_module.val_metric4.reset()
        pl_module.val_metric5.reset()

    
def train(train_dataset, trainable_model, config, test_function, val_dataset):
    # Initialize
    is_main_process, rank = get_rank() == 0, get_rank()
    torch.cuda.set_device(rank)
    # config = get_config()

    training_config = config["train"]
    run_name = time.strftime("%Y%m%d-%H%M%S")

    # Initialize WanDB
    wandb_config = training_config.get("wandb", None)
    if wandb_config is not None and is_main_process:
        init_wandb(wandb_config, run_name)

    print("Rank:", rank)
    if is_main_process:
        print("Config:", config)

    # Initialize dataloader
    print("Dataset length:", len(train_dataset))
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.get("batch_size", 1),
        shuffle=True,
        num_workers=training_config["dataloader_workers"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=training_config["dataloader_workers"],
        drop_last=True,
    )

    # Callbacks for testing and saving checkpoints
    if is_main_process:
        callbacks = [TrainingCallback(run_name, training_config, test_function)]

    # Initialize trainer
    trainer = L.Trainer(
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=callbacks if is_main_process else [],
        enable_checkpointing=False,
        enable_progress_bar=True,
        logger=False,
        # max_steps=training_config.get("max_steps", -1),
        # max_epochs=training_config.get("max_epochs", -1),
        max_steps = 100,
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
        # check_val_every_n_epoch=1,
        val_check_interval=10,
    )

    setattr(trainer, "training_config", training_config)
    setattr(trainable_model, "training_config", training_config)

    # Save the training config
    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}")
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)

    # Start training
    # if is_main_process:
    #     print("Initial validation (debug before fit)")
    #     trainer.validate(trainable_model, dataloaders=val_loader)
    # trainer.test(trainable_model, val_loader)

    trainer.fit(trainable_model, train_loader,val_dataloaders = val_loader)