# import necessary libraries
import hydra

from src import FluoresenceReg, ISLES2024Dataset

@hydra.main(version_base=None, config_name='register_local', config_path='configs')
def main(config):
    # get data
    dataset = ISLES2024Dataset(config.data)
    id_dict = dataset[config.data.index]

    optimizer = FluoresenceReg(config.model, id_dict)
    optimizer.fit()

if __name__ == "__main__":
    main()