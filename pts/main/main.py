import hydra
from omegaconf import DictConfig

from pts.gen import generate
from pts.models import MaskRGNetwork


@hydra.main(config_path="conf", config_name="main")
def run(cfg: DictConfig):
    if cfg.mode == "gen":
        generate(cfg.gen)

    if cfg.mode == "train_rg":
        rg_net = MaskRGNetwork(cfg.reward_model)
        rg_net.train_model()


if __name__ == "__main__":
    run()
