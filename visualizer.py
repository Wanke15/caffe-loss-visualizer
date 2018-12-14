'''
Usage: 
python visualizer.py --log test_log.txt --output test_loss.png
'''

import re
import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot(log_file, fig_output):
	file_format = fig_output.split('.')[-1]
	assert file_format in ['png', 'jpg'], "Supported output file format '.png, .jpg', but got: '%s'" % file_format
	with open(log_file, 'r', encoding='utf-8') as f:
		logs = f.readlines()


	losses = []
	is_caffe_log = False
	for l in logs:
		m = re.search('Iteration \d+, loss = \d+.\d+', l)
		if m:
			title = 'Training Loss'
			losses.append(float(m[0].split(' ')[-1]))
			is_caffe_log = True
		else:
			m = re.search('Batch \d+, loss = \d+.\d+', l)
			if m:
				title = 'Test Loss'
				losses.append(float(m[0].split(' ')[-1]))
				is_caffe_log = True
			
	if not is_caffe_log:
		print('The file specified may be not Caffe log file!')
		exit()

	losses = np.array(losses)

	plt.title(title)
	plt.plot(losses)
	plt.xlabel('Iter')
	plt.ylabel('Loss')
	plt.savefig(fig_output)
	plt.show()
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(
            description='Visualize caffe logs')

	parser.add_argument(
            '--log', default="log.txt", type=str,
            help='log file')
	
	parser.add_argument(
            '--output', default="loss.png", type=str,
            help='save plot output')

	args = parser.parse_args()
	
	plot(args.log, args.output)



