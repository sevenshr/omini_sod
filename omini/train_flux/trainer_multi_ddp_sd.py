import lightning as L
from diffusers.pipelines import StableDiffusion3Pipeline
import torch
import wandb
import os
import yaml
from peft import LoraConfig, get_peft_model_state_dict
from torch.utils.data import DataLoader
import time

from typing import List

import prodigyopt

from ..pipeline.sd_omini import transformer_forward, encode_images
import utils
import torch.nn.functional as F
from ..pipeline.sd_omini import Condition, convert_to_condition, generate
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
        sd_pipe_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        adapter_names: List[str] = [None, None, "default"],
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        val_datasets_name: List[str] = [],
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config

        # Load the sd pipeline
        self.sd_pipe: StableDiffusion3Pipeline = StableDiffusion3Pipeline.from_pretrained(
            sd_pipe_id, torch_dtype=dtype
        ).to(device)
        self.transformer = self.sd_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()

        # Freeze the sd pipeline
        self.sd_pipe.text_encoder.requires_grad_(False).eval()
        self.sd_pipe.text_encoder_2.requires_grad_(False).eval()
        self.sd_pipe.text_encoder_3.requires_grad_(False).eval()
        self.sd_pipe.vae.requires_grad_(False).eval()
        self.adapter_names = adapter_names
        self.adapter_set = set([each for each in adapter_names if each is not None])

        # Initialize LoRA layers
        self.lora_layers = self.init_lora(lora_path, lora_config)
        self.to(device).to(dtype)

        self.val_metrics = {}
        for dataset_name in val_datasets_name:
            self.val_metrics[dataset_name] =  torch.nn.ModuleDict({
                "m1": utils.AveragerDDP(use_device=device),
                "m2": utils.AveragerDDP(use_device=device),
                "m3": utils.AveragerDDP(use_device=device),
                "m4": utils.AveragerDDP(use_device=device),
                "m5": utils.AveragerDDP(use_device=device),
            })
        self.metric_fn = utils.calc_cod_multi


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
            self.transformer.set_adapter(self.adapter_set)
            # TODO: Check if this is correct (p.requires_grad)
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        return list(lora_layers)

    def save_lora(self, path: str):
        for adapter_name in self.adapter_set:
            StableDiffusion3Pipeline.save_lora_weights(
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
        batch, dataset_names = batch   # ⭐ 新增
        dataset_name = dataset_names[0]

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
            x_0 = encode_images(self.sd_pipe, imgs)

            # Prepare text input
            (
                prompt_embeds,
                _,
                pooled_prompt_embeds, _
            ) = self.sd_pipe.encode_prompt(
                prompt=prompts,
                prompt_2=None,
                prompt_3=None,
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                device=self.sd_pipe.device,
                num_images_per_prompt=1,
                max_sequence_length=self.model_config.get("max_sequence_length", 512),
                lora_scale=None,
            )
            # Prepare t and x_t
            # t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            t = torch.ones((imgs.shape[0],), device=self.device)
            x_1 = torch.randn_like(x_0).to(self.device)
            t_ = t.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)
            if image_latent_mask is not None:
                x_0 = x_0[:, image_latent_mask[0]]
                x_1 = x_1[:, image_latent_mask[0]]
                x_t = x_t[:, image_latent_mask[0]]

            # Prepare conditions
            condition_latents, condition_ids = [], []
            for cond, p_delta, p_scale, latent_mask in zip(
                conditions, position_deltas, position_scales, latent_masks
            ):
                # Prepare conditions
                c_latents = encode_images(self.sd_pipe, cond)
                # Append to the list
                if latent_mask is not None:
                    c_latents = c_latents[latent_mask]
                condition_latents.append(c_latents)

        branch_n = 2 + len(conditions)
        group_mask = torch.ones([branch_n, branch_n], dtype=torch.bool).to(self.device)
        # Disable the attention cross different condition branches
        group_mask[2:, 2:] = torch.diag(torch.tensor([1] * len(conditions)))
        # Disable the attention from condition branches to image branch and text branch
        if self.model_config.get("independent_condition", False):
            group_mask[2:, :2] = False

        # Forward pass
        if len(self.adapter_names)>4:
            if prompts == "RGB and thermal to mask":
                use_adapters = self.adapter_names[:3] + self.adapter_names[-1] 
            else:
                use_adapters = self.adapter_names[:branch_n]
        else:
            use_adapters = self.adapter_names[:branch_n]
        
        transformer_out = transformer_forward(
            self.transformer,
            image_features=[x_t, *(condition_latents)],
            text_features=[prompt_embeds],
            # There are three timesteps for the three branches
            # (text, image, and the condition)
            timesteps=[t, t] + [torch.zeros_like(t)] * len(conditions),
            # Same as above
            pooled_projections=[pooled_prompt_embeds] * branch_n,
            # The LoRA adapter names of each branch
            adapters=use_adapters,
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

        self.log(
        "train_loss",
        step_loss,
        prog_bar=True,
        on_step=True,
        on_epoch=False,
        sync_dist=True,
        batch_size=imgs.shape[0],
        )
        return step_loss

    def generate_a_sample(self):
        raise NotImplementedError("Generate a sample not implemented.")


    def test_step(self, batch, batch_idx, dataloader_idx):
        dataset_name = batch["dataset"][0]  # ⭐ 新增
        imgs, prompts = batch["image"], batch["description"]
        use_device = imgs.device
        bs = batch["image"].shape[0]
        image_latent_mask = batch.get("image_latent_mask", None)
        conditions_use, conditions, position_deltas, position_scales, latent_masks = [], [], [], [], []
        if len(self.adapter_names)>4 and prompts == "RGB and thermal to mask":
            use_adapters = self.adapter_names[:3] + self.adapter_names[-1] 
        else:
            use_adapters = self.adapter_names

        for i in range(1000):
            if f"condition_{i}" not in batch:
                break
            conditions.append(batch[f"condition_{i}"])
            position_deltas.append(batch.get(f"position_delta_{i}", [0, 0]))
            position_scales.append(batch.get(f"position_scale_{i}", [1.0])[0])
            latent_masks.append(batch.get(f"condition_latent_mask_{i}", None))
            conditions_use.append(Condition(conditions[-1], use_adapters[2+i], position_deltas[-1][0], position_scales[-1]))
        generator = torch.Generator(device=self.device)
        generator.manual_seed(42)
        pre_res = generate(
            self.sd_pipe,
            prompt=prompts,
            conditions=conditions_use,
            height=batch["condition_0"].shape[-1],
            width=batch["condition_0"].shape[-2],
            generator=generator,
            model_config=self.model_config,
            kv_cache= False,
            output_type = "pt",
            num_inference_steps = 1,
        )
        res = pre_res.images.mean(dim=1).unsqueeze(1)
        # import ipdb
        # ipdb.set_trace()
        res = F.interpolate(res, size=imgs.shape[-2:], mode='bilinear', align_corners=False)
        res = res.cpu().float().numpy().squeeze(1)
        imgs = imgs.cpu().float().numpy().squeeze(1)
        # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        result1, result2, result3, result4, result5 = self.metric_fn(res, imgs)

        metrics = self.val_metrics[dataset_name]
        metrics["m1"].update(torch.tensor(result1.item()).to(use_device), bs)
        metrics["m2"].update(torch.tensor(result2.item()).to(use_device), bs)
        metrics["m3"].update(torch.tensor(result3.item()).to(use_device), bs)
        metrics["m4"].update(torch.tensor(result4.item()).to(use_device), bs)
        metrics["m5"].update(torch.tensor(result5.item()).to(use_device), bs)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        return self.test_step(batch, batch_idx, dataloader_idx)

    def on_train_epoch_start(self):
        dataloader = self.trainer.train_dataloader

        # Lightning 兼容（list / CombinedLoader）
        if isinstance(dataloader, list):
            dataloader = dataloader[0]

        if hasattr(dataloader, "batch_sampler"):
            dataloader.batch_sampler.set_epoch(self.current_epoch)

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
        if torch.is_tensor(outputs):
            loss_value = outputs.item() * trainer.accumulate_grad_batches
        elif isinstance(outputs, dict) and "loss" in outputs:
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
        else:
            loss_value = None

        if trainer.is_global_zero and self.use_wandb and loss_value is not None:
            wandb.log({
                "step": batch_idx,
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
                "loss": loss_value,
                "t": getattr(pl_module, "last_t", 0.0),
            })


        if trainer.is_global_zero and self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        # Save LoRA weights at specified intervals
        if trainer.is_global_zero and self.total_steps % self.save_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights"
            )
            pl_module.save_lora(
                f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            )

    def on_validation_epoch_end(self, trainer, pl_module):

        save_path = f"{self.save_path}/{self.run_name}/output"
        file_name = f"lora_{self.total_steps}"

        if trainer.is_global_zero:
            os.makedirs(save_path, exist_ok=True)
            log_path = os.path.join(save_path, "val_metrics.txt")

        for dataset_name, metrics in pl_module.val_metrics.items():
            
            m1 = metrics["m1"].compute().detach().cpu().item()
            m2 = metrics["m2"].compute().detach().cpu().item()
            m3 = metrics["m3"].compute().detach().cpu().item()
            m4 = metrics["m4"].compute().detach().cpu().item()
            m5 = metrics["m5"].compute().detach().cpu().item()
            if trainer.is_global_zero:
                print(f"[{dataset_name}] maxfm:{m1:.4f} wfm:{m2:.4f} em:{m3:.4f} sm:{m4:.4f} mae:{m5:.4f}")

                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(f"{file_name} {dataset_name} | maxfm:{m1:.6f}, wfm:{m2:.6f}, em:{m3:.6f}, sm:{m4:.6f}, mae:{m5:.6f}\n")

        # Log to WandB
                if self.use_wandb:
                    wandb.log({
                        f"{dataset_name}_maxfm": m1,
                        f"{dataset_name}_wfm": m2,
                        f"{dataset_name}_em": m3,
                        f"{dataset_name}_sm": m4,
                        f"{dataset_name}_mae": m5,
                    })

            for m in metrics.values():
                m.reset()

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

    
def train(train_dataset, trainable_model, config, test_function, val_datasets, batch_sampler):
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
        batch_sampler = batch_sampler, 
        num_workers=training_config["dataloader_workers"],
        pin_memory=True,
        persistent_workers=training_config.get("dataloader_workers", 0) > 0,
    )

    val_loaders = []
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    for name, dataset in val_datasets.items():
        sampler = None
        if world_size > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=True,
            )
        loader = DataLoader(
            dataset,
            # batch_size=training_config.get("batch_size", 1),
            sampler=sampler,
            batch_size=1,
            shuffle=False,
            num_workers=training_config["dataloader_workers"],
            drop_last=True,
        )
        val_loaders.append(loader)

    # Callbacks for testing and saving checkpoints
    callbacks = [TrainingCallback(run_name, training_config, test_function)]

    # Initialize trainer
    trainer = L.Trainer(
        accelerator="gpu",
        devices=training_config.get("devices", torch.cuda.device_count()),
        strategy="ddp",
        use_distributed_sampler=False,  # 训练 sampler 由你自己管理
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=callbacks,
        enable_checkpointing=False,
        enable_progress_bar=True,
        logger=False,
        max_steps=training_config.get("max_steps", -1),
        # max_epochs=training_config.get("max_epochs", -1),
        # max_steps = 100,
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
        # check_val_every_n_epoch=1,
        val_check_interval=training_config.get("save_interval", -1),
    )

    setattr(trainer, "training_config", training_config)
    setattr(trainable_model, "training_config", training_config)

    # Save the training config
    save_path = training_config.get("save_path", "./output")
    if wandb_config is not None and is_main_process:
        os.makedirs(f"{save_path}/{run_name}")
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)

    # Start training
    # if is_main_process:
    #     print("Initial validation (debug before fit)")
    #     trainer.validate(trainable_model, dataloaders=val_loader)
    # trainer.test(trainable_model, val_loader)

    trainer.fit(trainable_model, train_loader,val_dataloaders = val_loaders)