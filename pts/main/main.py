import cv2 as cv
import hydra
from omegaconf import DictConfig

from pts.gen import generate
from pts.main import train_dqn
from pts.models import MaskRGNetwork


@hydra.main(config_path="conf", config_name="main")
def run(cfg: DictConfig):
    if cfg.mode == "gen":
        generate(cfg.env)

    if cfg.mode == "train_rg":
        rg_net = MaskRGNetwork(cfg.reward_model)
        rg_net.train_model()
        rg_net.save_model(cfg.reward_model.path_store_weights)

    if cfg.mode == "test_rg":
        rg_net = MaskRGNetwork(cfg.reward_model)
        img = cv.imread(cfg.test.path_test_image)
        rg_net.eval_single_img(img)

    if cfg.mode == "train_dqn":
        train_dqn(cfg.eval_model, cfg.reward_model, cfg.env)


if __name__ == "__main__":
    run()
