"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import util
from util import html
import ntpath

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

image_dir = os.path.join(opt.dataroot, 'synth')
if not os.path.isdir(image_dir):
	os.makedirs(image_dir)


for i, data_i in enumerate(dataloader):
	if i * opt.batchSize >= opt.how_many:
		break

	generated = model(data_i, mode='inference')

	img_path = data_i['path']
	for b in range(generated.shape[0]):
		print('process image... %s' % img_path[b])
		visuals = OrderedDict([('input_label', data_i['label'][b]), ('synthesized_image', generated[b])])
		visuals = visualizer.convert_visuals_to_numpy(visuals) 

		test =visuals.items()
		# for label, image_numpy in visuals.items():
		short_path = ntpath.basename(img_path[b:b + 1][0])
		name = os.path.splitext(short_path)[0]

		image_name = '%s.png' % (name)
		save_path = os.path.join(image_dir, image_name)
		util.save_image(visuals['synthesized_image'], save_path, opt.width, opt.height)
		# visualizer.save_images(webpage, visuals, img_path[b:b + 1])

# webpage.save()
