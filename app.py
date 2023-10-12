from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from image import Image
# from PIL import Image
 
app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def brighten(image, factor):
    x_pixels, y_pixels, num_channels = image.array.shape  # represents x, y pixels of image, # channels (R, G, B)
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)  # making a new array to copy values to!

    new_im.array = image.array * factor

    return new_im

def adjust_contrast(image, factor, mid):
    x_pixels, y_pixels, num_channels = image.array.shape  # represents x, y pixels of image, # channels (R, G, B)
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)  # making a new array to copy values to!
    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                new_im.array[x, y, c] = (image.array[x, y, c] - mid) * factor + mid

    return new_im

def blur(image, kernel_size):
    x_pixels, y_pixels, num_channels = image.array.shape 
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels) 
    neighbor_range = kernel_size // 2  
    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                total = 0
                for x_i in range(max(0,x-neighbor_range), min(new_im.x_pixels-1, x+neighbor_range)+1):
                    for y_i in range(max(0,y-neighbor_range), min(new_im.y_pixels-1, y+neighbor_range)+1):
                        total += image.array[x_i, y_i, c]
                new_im.array[x, y, c] = total / (kernel_size ** 2)
    return new_im

 
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print('upload_image filename: ' + filename)

        prototxt_path = 'models/colorization_deploy_v2.prototxt'
        model_path = 'models/colorization_release_v2.caffemodel'
        kernel_path = 'models/pts_in_hull.npy'
        image_path = f"static/uploads/{filename}"

        net = cv2.dnn.readNetFromCaffe(prototxt_path,model_path)
        points = np.load(kernel_path)

        points = points.transpose().reshape(2,313,1,1)
        net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
        net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1,313],2.606,dtype="float32")]

        bw_image = cv2.imread(image_path)
        normalized = bw_image.astype("float32") / 255.0
        lab = cv2.cvtColor(normalized,cv2.COLOR_BGR2LAB)

        resized = cv2.resize(lab,(224,224))
        L = cv2.split(resized)[0]
        L-=50

        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1,2,0))

        ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))
        L = cv2.split(lab)[0]

        colorized = np.concatenate((L[:,:,np.newaxis],ab),axis=2)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = (255.0 * colorized).astype("uint8")
        filename2 = "colorized_" + filename
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename2), colorized)

        print("Recolor done and uploaded")
        #===============================================================================
        lake = Image(filename=filename2)

        # brightening
        brightened_im = brighten(lake, 1.7)
        brightened_im.write_image('brightened.png')

        # darkening
        darkened_im = brighten(lake, 0.3)
        darkened_im.write_image('darkened.png')

        # increase contrast
        incr_contrast = adjust_contrast(lake, 2, 0.5)
        incr_contrast.write_image('increased_contrast.png')
        
        # decrease contrast
        decr_contrast = adjust_contrast(lake, 0.5, 0.5)
        decr_contrast.write_image('decreased_contrast.png')
        #===============================================================================
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()



