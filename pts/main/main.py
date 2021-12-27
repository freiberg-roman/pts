import hydra
from omegaconf import DictConfig

from pts.gen import generate


@hydra.main(config_path="conf", config_name="main")
def run(cfg: DictConfig):
    generate(cfg.gen)


if __name__ == "__main__":
    run()
