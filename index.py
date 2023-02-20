!pip install pillow

from PIL import Image
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageFilter

from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')

display(js)

data = eval_js('takePhoto({})'.format(quality))

binary = b64decode(data.split(',')[1])

with open(filename, 'wb') as f:
  f.write(binary)
return filename

from IPython.display import Image

try:
  filename = take_photo()
  print('Saved to {}'.format(filename))
 
  display(Image(filename))
except Exception as err:
  print(str(err))

from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

Im = Image.open("photo.jpg")
Draw=ImageDraw.Draw(Im)
ttfont=ImageFont.truetype("the route of the font",100)

model = load_model('the route of machine learning modal (.h5)')

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

image = Image.open("photo.jpg")

size = (224, 224)

image = ImageOps.fit(image, size, Image.ANTIALIAS)

image_array = np.asarray(image)

normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

data[0] = normalized_image_array

prediction = model.predict(data)

maxP = np.argmax(prediction)
maxP2 = np.max(prediction)

if prediction.max()>0.5:  
    if maxP==0 :
        print("人員未配戴口罩")
        Draw.text((0,0),"沒口罩",fill ="#E0543F",font=ttfont)
        display(Im)              
    elif maxP==1 :
        print("人員有配戴口罩")
        Draw.text((0,0),"有口罩",fill ="#3FE07D",font=ttfont)
        display(Im)
    elif maxP==2 :
        print("背景畫面未有人物可辨識")
        display(Im)
else:
    print('無法辨識畫面')

print(f"{maxP2*100:.2f}%")

print(prediction)

display(image)
