import os
from sys import argv, exit
import numpy as np

if __name__ == '__main__':
	srcDir = argv[1]
	trgDir = argv[2]

	files = os.listdir(srcDir)
	leng = len(files)

	trgLen = 100
	for i in range(trgLen):
		idx = np.random.randint(leng)

		fileName = os.path.join(srcDir, files[idx])

		# cmd = "cp {} {}".format(fileName, trgDir)
		cmd = "cp {} {}{}".format(fileName, trgDir, files[idx].replace(".jpg", "Z.jpg"))
		os.system(cmd)