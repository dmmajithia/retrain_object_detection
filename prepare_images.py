import argparse
import cv2
import time
import readchar
from PIL import Image
import numpy as np
from skimage.measure import compare_ssim
import imutils
from pickler import ImageData

ARGS = {}

def help():
	print('h=help, q=quit, n=next, p=previous, j=jump /{num/}')

def show_diffs(imA,imB):
	imageA = cv2.imread(ARGS.outputpath + imA)
	imageB = cv2.imread(ARGS.outputpath + imB)
	grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
	grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
	(score, diff) = compare_ssim(grayA, grayB, full=True)
	diff = (diff * 255).astype("uint8")
	# print("SSIM: {}".format(score))
	thresh = cv2.threshold(diff, 0, 255,
		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	boxes = [cv2.boundingRect(c) for c in cnts]
	max_area, max_box = 0.0, []
	for c in cnts:
		(x, y, w, h) = cv2.boundingRect(c)
		area = w*h
		if area > max_area:
			max_area = area
			max_box = [(x,y),(x + w, y + h)]
		# cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
		# cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
	if max_area>0:
		cv2.rectangle(imageA, max_box[0], max_box[1], (0, 0, 255), 2)
		cv2.rectangle(imageB, max_box[0], max_box[1], (0, 0, 255), 2)
	cv2.imshow("Base", imageA)
	cv2.imshow("Frame", imageB)
	cv2.waitKey(20)
	return max_box

def loop(lines):
	if len(lines) < 2:
		print('n < 1 !!! Exiting.')
		return
	base = 0
	frame = 1
	base_set = False
	auto_add = False

	i = -1
	ch = 'n'
	n = len(lines)
	data = ImageData(path=ARGS.outputpath)
	if ARGS.pickle_name != '':
		data = ImageData.load(pickle_name=ARGS.pickle_name)
		data.new(path=ARGS.outputpath)

	while ch != 'q':
		# print(ch)
		d,j,k = 0,0,0
		if ch == 'n':
			d = 1
		elif ch == 'p':
			d = -1
		elif ch == 'k':
			print('Enter jump k: ')
			k = int(input())
		elif ch == 'j':
			print('Jump to: ')
			j = int(input())
		elif ch == 'h':
			help()
		elif ch == 'e':
			auto_add = True
			print('Enabled auto add.')
		elif ch == 'd':
			auto_add = False
			print('Disabled auto add.')
		elif ch == 'a':
			print('Added to dataset.')
		elif ch == 'r':
			print('Removed from dataset.')
		elif ch == 'b':
			base_set = not base_set
			print('Base set.') if base_set else print('Base unset.')
			auto_add = auto_add and base_set
			print('Enabled auto add.') if auto_add else print('Disabled auto add.')
		if d != 0:
			if not base_set:
				if base+d in range(0,n):
					base += d
			else:
				if frame+d in range(0,n):
					frame += d
		elif k != 0:
			if not base_set:
				if base+k in range(0,n):
					base += k
			else:
				if frame+k in range(0,n):
					frame += k
		elif j != 0 and j in range(0,n):
			if not base_set:
				base = j
			else:
				frame = j
		print(f"{ch}, base={base}, frame={frame}, auto_add={auto_add}, base_set={base_set}")
		box = show_diffs(lines[base].rstrip(),lines[frame].rstrip())
		if (auto_add and ch!='r') or ch=='a':
			print('will add')
			if box != []:
				data.add(lines[frame].rstrip(), box)
		if ch=='r':
			print('will remove')
			data.remove(lines[frame].rstrip())

		ch = readchar.readchar()
	if ARGS.pickle_name == '':
		print('Enter new pickle name:')
		ARGS.pickle_name = input()
	data.save(pickle_name=ARGS.pickle_name)
	cv2.destroyAllWindows()

if __name__ == "__main__":
	PARSER = argparse.ArgumentParser(description='Test gstreamer.')
	PARSER.add_argument('-o', '--outputpath', action='store', default='/Users/dhawalmajithia/Desktop/works/aqi/output/', help="Save frames here.")
	PARSER.add_argument('-if', '--inputfile', action='store', default='none', help="Method.")
	PARSER.add_argument('-p', '--pickle_name', action='store', default='', help="Method.")
	ARGS = PARSER.parse_args()
	
	lines = []
	with open(ARGS.inputfile,'r') as f:
		lines=f.readlines()
	loop(lines=lines)



