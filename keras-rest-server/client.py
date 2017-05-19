from __future__ import print_function
import requests
import sys
import numpy as np
from sklearn.preprocessing import normalize
from argparse import ArgumentParser
import time
# from keras.preprocessing import image as kerasimage

import base64
from cStringIO import StringIO
import json
import sys
import time

parser = ArgumentParser()
parser.add_argument('--server', type=str,dest='server')
FLAGS = parser.parse_args()



try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

def img_to_array(img):
    x = np.asarray(img, dtype='float32')
    return x


def load_img(path, grayscale=False, target_size=None):
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size:
        hw_tuple = (target_size[1], target_size[0])
        if img.size != hw_tuple:
            img = img.resize(hw_tuple)
    return img    

def topN(index,N):
	index = np.expand_dims(index,axis=0)
	index_normed = normalize(index,axis=1)
	index_top = np.argsort(-index_normed)[:,:N]
	return index_top[0]


def make_request_json(input_image):
	image = load_img(input_image, target_size=(224, 224))
	x = img_to_array(image)
	### Code to convert the image to base 64. Dont use the above line in that case
	# resized_handle = StringIO()
	# image.save(resized_handle, format='JPEG')
	# encoded_contents = base64.b64encode(resized_handle.getvalue())
	# resized_handle.close()	
	return x  

t1 = time.time()

sample_image = "/home/kicompute/Downloads/NewSampleImages/womenKurtis/110561ab_1.jpeg"
url = "http://localhost:7171/predict"

# Load Image and Resize and send image / image array
data = make_request_json(sample_image)
data = data.tolist()
# r = requests.post(url, json={'X_input':data})
headers = {'Content-Type': 'application/json', 'Accept':'application/json'}
r = requests.post(url,data=json.dumps(data), headers=headers)

# Sending Image URL instead of Image
# sample_image = "https://www.planwallpaper.com/static/images/9-credit-1.jpg"
# r = requests.post(url, json={'input':sample_image})

print(r.status_code, r.reason)
resp = r.json()
print(np.array(resp['pred_val'][0]))
t2 = time.time()
print("time taken",str(t2-t1))

