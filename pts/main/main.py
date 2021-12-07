import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="main")
def run(cfg: DictConfig):
    pass


if __name__ == "__main__":
    run()
