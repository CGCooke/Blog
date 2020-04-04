from PIL import Image
import glob

path = '2020-4-3-The-Exodus/*'
f_list = glob.glob(path)

for f in f_list:
	img = Image.open(f)
	height = int(img.size[1] / (img.size[0]/1400))
	if img.size[0]>1400:
		img = img.resize((1400,height),Image.LANCZOS)
	
	try:
		img.save(f[:-4]+'.jpg')
	except:
		img = img.convert("RGB")
		img.save(f[:-4]+'.jpg')