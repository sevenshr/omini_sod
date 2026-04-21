import time

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
import random
import numpy as np

from PIL import Image, ImageDraw

from datasets import load_dataset
import utils

from .trainer_multi import OminiModel, get_config, train
from ..pipeline.flux_omini import Condition, convert_to_condition, generate


from ..dataset_sod.data_multi import RGBSalObjDataset, RGBTSalObjDataset, RGBDSalObjDataset, MultiDataset, MultiSampler, test_dataset, SalObjDataset_val
import cv2
import torch.nn.functional as F
import wandb


@torch.no_grad()
def test_function(model, save_path, file_name):
    condition_size = model.training_config["dataset"]["condition_size"]
    target_size = model.training_config["dataset"]["target_size"]

    position_delta = model.training_config["dataset"].get("position_delta", [0, 0])
    position_scale = model.training_config["dataset"].get("position_scale", 1.0)

    adapter = model.adapter_names[2]
    condition_type = model.training_config["condition_type"]
    test_list = []

    
    os.makedirs(save_path, exist_ok=True)
    image_root = model.training_config["val_dataset"]["image_root"]
    gt_root = model.training_config["val_dataset"]["gt_root"]
    depth_root = model.training_config["val_dataset"]["depth_root"]
    test_loader = test_dataset(image_root, gt_root, depth_root, model.training_config["val_dataset"]["valsize"])

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()
    val_metric5 = utils.Averager()
    metric_fn = utils.calc_cod_single

    for i in range(test_loader.size):
        image, gt, depth, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.cuda()
        generator = torch.Generator(device=model.device)
        generator.manual_seed(42)
        prompt = "give the saliency map of the image"
        condition = Condition(image[0], adapter, position_delta, position_scale)

        pre_res = generate(
            model.flux_pipe,
            prompt=prompt,
            conditions=[condition],
            height=target_size[1],
            width=target_size[0],
            generator=generator,
            model_config=model.model_config,
            kv_cache=model.model_config.get("independent_condition", False),
            output_type = "pt",
            num_inference_steps = 1,
        )
        res = pre_res.images[0].mean(dim=0).unsqueeze(0).unsqueeze(0)
        # import ipdb
        # ipdb.set_trace()
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().cpu().float().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        result1, result2, result3, result4, result5 = metric_fn(res, gt)
        val_metric1.add(result1.item(), 1)
        val_metric2.add(result2.item(), 1)
        val_metric3.add(result3.item(), 1)
        val_metric4.add(result4.item(), 1)
        val_metric5.add(result5.item(), 1)

        # print('save img to: ', save_path + name.split('/')[-1])
        # file_path = os.path.join(save_path,file_name, name.split('/')[-1])
        # cv2.imwrite(save_path + file_path, res * 255)
    m1 = val_metric1.item()
    m2 = val_metric2.item()
    m3 = val_metric3.item()
    m4 = val_metric4.item()
    m5 = val_metric5.item()

    print('metric1: {:.4f}'.format(m1))
    print('metric2: {:.4f}'.format(m2))
    print('metric3: {:.4f}'.format(m3))
    print('metric4: {:.4f}'.format(m4))
    print('metric5: {:.4f}'.format(m5))

    log_path = os.path.join(save_path, 'val_metrics.txt')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"{file_name} metric1: {m1:.6f}, metric2: {m2:.6f}, metric3: {m3:.6f}, metric4: {m4:.6f}, metric5: {m5:.6f}\n")

    # Log to WandB
    try:
        wandb.log({
            "val_metric1": m1,
            "val_metric2": m2,
            "val_metric3": m3,
            "val_metric4": m4,
            "val_metric5": m5,
        })
    except Exception as e:
        print("Failed to initialize WanDB:", e)


def main():
    # Initialize
    config = get_config()
    training_config = config["train"]
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    # Load dataset text-to-image-2M
    datasets = {
        'DUTS-TR': RGBSalObjDataset(datasets=['DUTS-TR'], imgsize=training_config["train_dataset"]["trainsize"], mode='train'),
        'train_DUT': RGBDSalObjDataset(datasets=['train_DUT'], imgsize=training_config["train_dataset"]["trainsize"], mode='train'),
        'VT_train': RGBTSalObjDataset(datasets=['VT_train'], imgsize=training_config["train_dataset"]["trainsize"], mode='train'),
        }

    multi_dataset = MultiDataset(datasets)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        world_size = 1
        rank = 0

    batch_sampler = MultiSampler(
        datasets,
        batch_size=training_config.get("batch_size", 1),
        rank=rank,
        world_size=world_size,
        # weights=training_config.get("dataset_weights", None),
    )

    # Initialize custom dataset
    # dataset = SalObjDataset(
    #     image_root=training_config["train_dataset"]["image_root"],
    #     gt_root=training_config["train_dataset"]["gt_root"],
    #     depth_root=training_config["train_dataset"]["depth_root"],
    #     trainsize=training_config["train_dataset"]["trainsize"],
    #     condition_size=training_config["dataset"]["condition_size"],
    #     target_size=training_config["dataset"]["target_size"],
    #     condition_type=training_config["condition_type"],
    #     drop_text_prob=training_config["dataset"]["drop_text_prob"],
    #     drop_image_prob=training_config["dataset"]["drop_image_prob"],
    #     position_scale=training_config["dataset"].get("position_scale", 1.0),
    # )
    val_datasets = {
        'DUTS-TE': RGBSalObjDataset(datasets=['DUTS-TE'], imgsize=training_config["train_dataset"]["trainsize"], mode='test'),
        # 'DUT-OMRON': RGBSalObjDataset(datasets=['DUT-OMRON'], imgsize=training_config["train_dataset"]["trainsize"], mode='test'),
        'VT5000': RGBTSalObjDataset(datasets=['VT5000'], imgsize=training_config["train_dataset"]["trainsize"], mode='test'),
        # 'VT821': RGBTSalObjDataset(datasets=['VT821'], imgsize=training_config["train_dataset"]["trainsize"], mode='test'),
        'NJU2K': RGBDSalObjDataset(datasets=['NJU2K'], imgsize=training_config["train_dataset"]["trainsize"], mode='test'),
        # 'NLPR': RGBDSalObjDataset(datasets=['NLPR'], imgsize=training_config["train_dataset"]["trainsize"], mode='test'),
        }

    val_datasets_name = list(val_datasets.keys())
    # Initialize model
    trainable_model = OminiModel(
        flux_pipe_id=config["flux_path"],
        lora_config=training_config["lora_config"],
        device=f"cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        val_datasets_name=val_datasets_name,
        adapter_names=[None, None, *["default"] * 2],
    )

    train(multi_dataset, trainable_model, config, test_function, val_datasets, batch_sampler)


if __name__ == "__main__":
    main()
