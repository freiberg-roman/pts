import hydra
import torch
from omegaconf import DictConfig
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype
from torchvision.utils import draw_bounding_boxes

from pts.gen import generate
from pts.main import test_dqn, train_dqn
from pts.models import MaskRGNetwork
from pts.utils.ds_loader import PTSDataset
from pts.utils.image_helper import show
from pts.utils.merge_data import merge_data


@hydra.main(config_path="conf", config_name="main")
def run(cfg: DictConfig):
    if cfg.mode == "gen":
        generate(
            sessions=cfg.env.session_limit,
            save_to=cfg.data_base_path + cfg.process_id + "/",
        )

    if cfg.mode == "train_rg":
        rg_net = MaskRGNetwork(cfg.reward_model)
        if cfg.reward_model.train.use_pretrained:
            rg_net.load_weights()

        ds_loader = PTSDataset(cfg.reward_model.dataset)
        rg_net.set_data(ds_loader)
        rg_net.train_model()
        rg_net.save_model()

    if cfg.mode == "test_rg":
        rg_net = MaskRGNetwork(cfg.reward_model)
        rg_net.load_weights()

        img = read_image(cfg.reward_model.test.path_test_image)
        batch_int = torch.stack([img])
        batch = convert_image_dtype(batch_int, dtype=torch.float)

        out = rg_net.eval_single_img(batch)
        img_with_boxes = [
            draw_bounding_boxes(
                img, boxes=output["boxes"][output["scores"] > 0.1], width=4
            )
            for img, output in zip(batch_int, out)
        ]
        show(img_with_boxes)

    if cfg.mode == "train_dqn":
        train_dqn(cfg.eval_model, cfg.reward_model, cfg.env)

    if cfg.mode == "test_dqn":
        test_dqn(cfg.eval_model, cfg.env)

    if cfg.mode == "merge_data":
        merge_data(cfg.data_base_path)


if __name__ == "__main__":
    run()
