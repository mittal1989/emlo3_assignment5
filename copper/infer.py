from typing import Tuple, Dict

import lightning as L
import torch
import hydra
from omegaconf import DictConfig
from copper import utils
import random
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import requests
from io import BytesIO
import os

log = utils.get_pylogger(__name__)

@utils.task_wrapper
def infer(cfg: DictConfig) -> Tuple[dict, dict]:
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    # datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    # trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    object_dict = {
        "cfg": cfg,
        # "datamodule": datamodule,
        "model": model,
        # "trainer": trainer,
    }

    transform=T.Compose([
                    T.Resize((32, 32)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

    # Loading latest checkpoint from saved history
    latest_checkpoint_path = None
    for root, dirs, files in os.walk('./outputs/'):
        for file in files:
            if file.endswith(".ckpt"):
                if latest_checkpoint_path is None or file > latest_checkpoint_path:
                    latest_checkpoint_path = os.path.join(root, file)

    print("Latest_checkpoint_path:", latest_checkpoint_path)

    # Loading weights from latest checkpoint
    model = model.load_from_checkpoint(latest_checkpoint_path)

    if cfg.get("image_path"):
        # read web image of cat/dog
        response = requests.get(cfg.get("image_path"))
        image = Image.open(BytesIO(response.content))
    else:
    # Using random image from test data for inference
        dataset = ImageFolder(root="./data/PetImages_split/test/")

        # Selectig random index from test dataset
        indices = random.sample(range(len(dataset)), 1)[0]

        # image, label from selected index
        image, label = dataset[indices]

    # Preprocessing image
    image = transform(image)
    image = image.unsqueeze(0)

    # Predict the class of the image
    with torch.no_grad():
        # Prediction
        prediction = model(image)

        # Get the top 2 probabilities
        predicted_class = torch.softmax(prediction, dim=1)[0].numpy()

        dic = {'cat':round(predicted_class[0],2), 'dog': round(predicted_class[1],2)}
        print("Probability of cat vs dog:", dic)

    return prediction, object_dict

@hydra.main(version_base="1.3", config_path="../configs", config_name="infer.yaml")
def main(cfg: DictConfig):
    # train the model
    metric_dict, _ = infer(cfg)
    # prediction,_ = infer(cfg)

    return metric_dict


if __name__ == "__main__":
    main()
