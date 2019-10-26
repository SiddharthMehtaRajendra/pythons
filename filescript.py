import os
import shutil

rootbase = "/Users/sumeet/Desktop/DL/project/CK+/"
datadir = "cohn-kanade-images/"
dest = "unwanted/"

files = []
buf = 3

for root, directories, filenames in os.walk(rootbase + datadir):
	count = len(filenames) - buf
	for i, filename in enumerate(sorted(filenames)):
		if filename.endswith('.png'):
			if i < count:
				os.rename(root + "/" + filename, rootbase + dest +  filename)