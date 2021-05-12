#You may need to restart your runtime prior to this, to let your installation take effect
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
import numpy as np
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
import os
from detectron2.evaluation import COCOEvaluator
import logging
from detectron2.utils.visualizer import ColorMode


class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)

def inference(img_path,save_path,device = "cpu"):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")) #Get the basic model configuration from the model zoo 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    #Use the final weights generated after successful training for inference  
    cfg.MODEL.WEIGHTS = "model/model_final.pth"
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set the testing threshold for this model
    #Pass the validation dataset
    # cfg.DATASETS.TEST = ("pubdal6_val", )
    cfg.DATASETS.TRAIN = ("pubdal6_train",)
    cfg.DATASETS.TEST = ("pubdal6_val",)
    predictor = DefaultPredictor(cfg)

    # dataset_dicts = get_board_dicts("/content/val","val.json")
    # for d in random.sample(dataset_dicts, 10):    
    im = cv2.imread(img_path)
    outputs = predictor(im)
    print(outputs)
    board_metadata = MetadataCatalog.get("pubdal6_train").set(thing_classes=["Text","Title","List","Table","Figure"])
    v = Visualizer(im[:, :, ::-1],
                    metadata=board_metadata, 
                    scale=1,
                    instance_mode=ColorMode.IMAGE   
    )
    # from google.colab.patches import cv2_imshow
    import numpy as np
    from PIL import Image

    v = v.draw_instance_predictions(outputs["instances"].to("cpu")) #Passing the 
    data = v.get_image()[:, :, ::-1]
    img = Image.fromarray(data, 'RGB')
    img.save(save_path +  "/" + img_path.split("/")[-1])
  # img.show()
