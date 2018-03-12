from enum import Enum

class Models(Enum):
	bvlc_alexnet = "bvlc_alexnet.caffemodel"
	bvlc_googlenet = "bvlc_googlenet.caffemodel"
	bvlc_reference_caffenet = "bvlc_reference_caffenet.caffemodel"
	finetune_flickr_style = "finetune_flickr_style.caffemodel"
	ResNet18_ImageNet_CNTK_model = 5
	AlexNet_ImageNet_CNTK_model = 6
