import os
import numpy as np
#from sklearn.model_selection import train_test_split

# path for downloaded fashion images
root_fashion_dir = './deepfashion/fashion'
output_fashion_dir = './deepfashion/fashion_resize'
assert len(root_fashion_dir) > 0, 'please give the path of raw deep fashion dataset!'

input_images = {}
for f1 in os.listdir(root_fashion_dir):
	for f2 in os.listdir(os.path.join(root_fashion_dir, f1)):
		for f3 in os.listdir(os.path.join(root_fashion_dir, f1, f2)):
			for f4 in os.listdir(os.path.join(root_fashion_dir, f1, f2, f3)):
				input_images['fashion{}{}{}{}'.format(f1, f2, f3, f4)] = os.path.join(root_fashion_dir, f1, f2, f3, f4)

#train_images, test_images = train_test_split(input_images.keys, test_size=0.20, random_state=42)
train_images, test_images = input_images, input_images

train_path = os.path.join(output_fashion_dir,'train')
if not os.path.exists(train_path):
	os.mkdir(train_path)

test_path = os.path.join(output_fashion_dir, 'test')
if not os.path.exists(test_path):
	os.mkdir(test_path)
'''
print ('train images:')
x = 0
with open(output_fashion_dir + '\\train.lst', 'a') as the_file:
	for (n, f) in train_images.items():
		to_ = os.path.join(train_path, n)
		os.system('cp %s %s' %(f, to_))
		the_file.write(n + '\n')

		x+=1
		if x>10:
			break
'''
print ('test images:')
x = 0
with open(output_fashion_dir + '\\test.lst', 'a') as the_file:
	for (n, f) in test_images.items():
		to_ = os.path.join(test_path, n)
		os.system('cp %s %s' %(f, to_))
		#the_file.write(n + '\n')

		# x+=1
		# if x>10:
		# 	break

print ('finish')