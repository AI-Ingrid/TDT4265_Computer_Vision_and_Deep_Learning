from tops.config import instantiate, LazyConfig
from ssd import utils
from tqdm import tqdm
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def get_config(config_path):
    cfg = LazyConfig.load(config_path)
    cfg.train.batch_size = 1
    return cfg


def get_dataloader(cfg, dataset_to_visualize):
    if dataset_to_visualize == "train":
        # Remove GroundTruthBoxesToAnchors transform
        cfg.data_train.dataset.transform.transforms = cfg.data_train.dataset.transform.transforms[:-1]
        data_loader = instantiate(cfg.data_train.dataloader)
    else:
        cfg.data_val.dataloader.collate_fn = utils.batch_collate
        data_loader = instantiate(cfg.data_val.dataloader)

    return data_loader

def analyze_something(dataloader, cfg):

    """
    TODO: 
    [X] Lage en dataframe som ser sånn her ut
    +-------+-------+
    | Label | Boxes |
    +-------+-------+
    
    [] Lage en kolonne til som heter size

    Begynne å analysere dataframe-en
    [] Type of task -> Type of diagram (x,y) 
    [] Size of boxes of given the same label -> Box plot (size, labels)
    [] No. of observations given a label in general -> Bar diagram (labels, amount)
    """

    # Creating a dataframe to store the data
    data_frame = pd.DataFrame()
    for batch in tqdm(dataloader):
        for labels, boxes in zip(batch["labels"], batch["boxes"]):
            for label, box in zip(labels, boxes):
                data_frame = data_frame.append({"Label": label.numpy(), 
                                                "x1": box[0].numpy(), 
                                                "y1": box[1].numpy(), 
                                                "x2": box[2].numpy(), 
                                                "y2": box[3].numpy(),
                                                }, ignore_index=True)
    # Create a new column with the size of each boxes   
    data_frame["size"] = data_frame.apply(lambda row: (row["x2"] - row["x1"]) * (row["y2"] - row["y1"]), axis=1)
    print(data_frame.head())



def main():
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"
    
    # Printing out possible labels
    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)
    analyze_something(dataloader, cfg)


if __name__ == '__main__':
    main()
