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
    [] Lage en dataframe som ser sånn her ut
    +-------+--------+-------+-------+
    | Label | Height | Width | Boxes |
    +-------+--------+-------+-------+
    
    Begynne å analysere dataframe-en
    [] Type of task -> Type of diagram (x,y) 
    [] Size of boxes of given the same label -> Box plot (size, labels)
    [] No. of observations given a label in general -> Bar diagram (labels, amount)
    """
    data_frame = pd.DataFrame()
    for batch in tqdm(dataloader):
        data_frame = data_frame.append(batch, ignore_index=True)
        print("The keys in the batch are:", batch.keys())
        print('Labels: ',batch["labels"])
        
        # Size of the frame
        print('Height: ',batch["height"])
        print("Width: ",batch["width"])
        
        # Boxes in frame
        print("Boxes: ", batch["boxes"])
        print(type(batch))
        
        # Calculate size of a an object
        # box = [x1 y1 x2 y2]
        for box, label in zip(batch["boxes"], batch["labels"]):
            print(box)
            
            # First corner
            x1 = box[0]
            y1 = box[1]
            
            # Second corner
            x2 = box[2]
            y2 = box[3]
            
            box_width = x2 - x1
            box_height = y2 - y1
            
            box_size = box_width * box_height
  
        #exit()
    data_frame.show()


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
