from enum import Enum

class ModelsPrototxtURL(Enum):
	bvlc_alexnet = "http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel"
	bvlc_alexnet_prototxt = "https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/deploy.prototxt"

	bvlc_googlenet = "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel"
	bvlc_googlenet_prototxt = "https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt"

	bvlc_reference_caffenet = "http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel"
	bvlc_reference_caffenet_prototxt = "https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_reference_caffenet/deploy.prototxt"

	finetune_flickr_style = "http://dl.caffe.berkeleyvision.org/finetune_flickr_style.caffemodel"
	finetune_flickr_style_prototxt = "https://raw.githubusercontent.com/BVLC/caffe/master/models/finetune_flickr_style/deploy.prototxt"

	ResNet18_ImageNet_CNTK_model = 5
	AlexNet_ImageNet_CNTK_model = 6
