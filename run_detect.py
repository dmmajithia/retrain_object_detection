import argparse
import cv2
from edgetpu.detection.engine import DetectionEngine
import time
import readchar
from PIL import Image
import numpy as np
from skimage.measure import compare_ssim
import imutils
from pickler import ImageData as p


def get_detects(img):
	detects = engine.detect_with_image(img, threshold=ARGS.conf, top_k=10, keep_aspect_ratio=True, relative_coord=False)
	detects = [d for d in detects if d.label_id == 6]
	return detects


def loop(args,lines,engine):






if __name__ == "__main__":
	PARSER = argparse.ArgumentParser(description='Test gstreamer.')
	PARSER.add_argument('-m', '--model', action='store', default='/Users/dhawalmajithia/Desktop/works/aqi/tflite/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite', help="Path to detection model.")
	PARSER.add_argument('-l', '--label', action='store', default='/Users/dhawalmajithia/Desktop/works/aqi/tflite/coco_labels.txt', help="Path to labels text file.")
	PARSER.add_argument('-o', '--outputpath', action='store', default='/Users/dhawalmajithia/Desktop/works/aqi/output/', help="Save frames here.")
	PARSER.add_argument('-c', '--conf', type=int, action='store', default=30, help="Detect threshold")
	PARSER.add_argument('-if', '--inputfile', action='store', default='none', help="Method.")
	ARGS = PARSER.parse_args()
	ARGS.conf = ARGS.conf/100.00
	lines = []
	engine = DetectionEngine(ARGS.model)
	if ARGS.inputfile != 'none':
		with open(ARGS.inputfile,'r') as f:
			lines=f.readlines()
	loop(args=ARGS,lines=lines,engine=engine)