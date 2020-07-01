from PIL import Image
import os

img_dir = './deepfashion/fashion_resize/test_small/'
save_dir = './deepfashion/fashion_resize/test/'

if not os.path.exists(save_dir):
	os.mkdir(save_dir)

cnt = 0

all_files = os.listdir(img_dir)
all_files_len = len(all_files)
for item in all_files:
	if not item.endswith('.jpg') and not item.endswith('.png'):
		continue
	cnt = cnt + 1
	print('{}/{} ...'.format(cnt, all_files_len))
	img = Image.open(os.path.join(img_dir, item))
	imgcrop = img.crop((40, 0, 176+40, 256))
	imgcrop.save(os.path.join(save_dir, item))
