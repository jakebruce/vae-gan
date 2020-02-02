#===============================================================================
# MIT License
#
# Copyright (c) 2017 Jake Bruce
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#===============================================================================

import numpy      as np
import tensorflow as tf
import sys, cv2, itertools, time

sys.dont_write_bytecode = True # keep python from creating .pyc file clutter
from model import VAEGAN # imported from model.py

#===============================================================================
# PARAMS

INP_SIZE       = 784
ENC_SIZES      = [1024, 256, 64]
LAT_SIZE       = 2
DEC_SIZES      = list(reversed(ENC_SIZES))
ADV_SIZES      = ENC_SIZES

DROP_P         = 0.9
NOISE_P        = 0.9
LEARN_RATE     = 1e-3
LEARN_RATE_DEC = 0.9
ADV_LOSS       = 1e1
LAT_LOSS       = 1e0
REC_LOSS       = 1e0
MAX_GRADIENT   = 1e0

BATCH_SIZE     = 1000

MANIFOLD_GRID  = 10
VIZ_SIZE_X     = 1920/2
VIZ_SIZE_Y     = 1080/2
SAVE_FRAMES    = False

#===============================================================================
# HELPERS

class Timer:
    def __init__ (self, label): self.label = label
    def __enter__(self):        self.start = time.time()
    def __exit__ (self,a,b,c):  print "%s: %fs" % (self.label, time.time()-self.start)

def gaussian_limits(latent, N):
    xmn = np.median(latent[:,0])-N*latent[:,0].std(); xmx = np.median(latent[:,0])+N*latent[:,0].std()
    ymn = np.median(latent[:,1])-N*latent[:,1].std(); ymx = np.median(latent[:,1])+N*latent[:,1].std()
    return xmn, xmx, ymn, ymx

#===============================================================================
# DATA

def get_data():
    from tensorflow.examples.tutorials.mnist import input_data as mnist_input_data
    return mnist_input_data.read_data_sets("MNIST_data/", one_hot=False)

#===============================================================================
# SETUP

cv2.namedWindow("embeddings", cv2.WINDOW_NORMAL)
cv2.namedWindow("manifold",   cv2.WINDOW_NORMAL)
cv2.namedWindow("inspector",  cv2.WINDOW_NORMAL)
colors = [(255,0,0),     # 0 - blue
          (0,255,0),     # 1 - light green
          (0,0,255),     # 2 - red
          (255,0,255),   # 3 - purple
          (0,0,128),     # 4 - dark red
          (128,128,0),   # 5 - cyan
          (0,0,0),       # 6 - black
          (128,128,128), # 7 - gray
          (0,128,0),     # 8 - dark green
          (0,128,255)]   # 9 - orange

def reset():
    global xe, ye, xmn, xmx, ymn, ymx, paused, drag
    with open("loss.dat", "w") as f: f.write("")
    xe = 0; ye = 0; xmn = -1; xmx = 1; ymn = -1; ymx = 1
    paused = False
    drag   = False

    # initialize all network variables
    sess.run(tf.global_variables_initializer())

# enable clicking to inspect embeddings
def inspect_location(event, x, y, flags, param):
    global xe, ye, drag
    if   event == cv2.EVENT_LBUTTONDOWN:
        drag = True
    elif event == cv2.EVENT_LBUTTONUP:
        drag = False
    elif event == cv2.EVENT_MOUSEMOVE and drag:
        xe = x/float(VIZ_SIZE_X)*(xmx-xmn) + xmn
        ye = y/float(VIZ_SIZE_Y)*(ymx-ymn) + ymn

cv2.setMouseCallback("embeddings", inspect_location)

sess = tf.Session()

# build model network
encoder = VAEGAN(INP_SIZE, ENC_SIZES, LAT_SIZE, DEC_SIZES, ADV_SIZES, DROP_P, NOISE_P,
                 LEARN_RATE, REC_LOSS, LAT_LOSS, ADV_LOSS, MAX_GRADIENT)

reset()

# get MNIST data
train, valid, test = get_data()
test_images        = np.array(list(valid.images)+list(test.images)+list(train.images[:1000]))
test_labels        = np.array(list(valid.labels)+list(test.labels)+list(train.labels[:1000]))

digit_colors = np.zeros((test_labels.shape[0], 3), dtype=np.uint8)
for d in range(10): digit_colors[test_labels==d,:] = colors[d]

#===============================================================================
# TRAIN

batch_id = 0
while True:
    if not paused:
        #----------
        # TRAINING
        #----------
        examples = np.random.randint(0,train.num_examples, size=(BATCH_SIZE))
        batch    = train.images[examples]

        # run input through network and do gradient descent update
        latent, recon, losses, adv_loss = encoder.train(sess, batch)

        print "Batch %s VAE losses: %s, adv loss: %s" % (batch_id, np.array(losses), adv_loss)
        with open("loss.dat", "a") as f: f.write("%s %s\n"%(batch_id," ".join([str(np.log(l+1e-18)) for l in losses+[adv_loss]])))

        if batch_id > 0 and batch_id % 100 == 0: sess.run(encoder.learn_rate.assign(encoder.learn_rate*LEARN_RATE_DEC))

        #---------
        # TESTING
        #---------

        # test on entire test set
        latent, recon, losses = encoder.test(sess, test_images)

        # visualize latent embeddings in 2d space
        emb_img = np.full((VIZ_SIZE_Y,VIZ_SIZE_X,3), 255, dtype=np.uint8)
        xmn, xmx, ymn, ymx = gaussian_limits(latent, 3)
        xmn *= float(VIZ_SIZE_X)/VIZ_SIZE_Y; xmx *= float(VIZ_SIZE_X)/VIZ_SIZE_Y
        latent_coords = (((latent-[xmn,ymn]) / [xmx-xmn+1e-18,ymx-ymn+1e-18])*[VIZ_SIZE_X,VIZ_SIZE_Y]).astype(np.int32)
        latent_coords[:,0] = np.clip(latent_coords[:,0], 0, VIZ_SIZE_X-1)
        latent_coords[:,1] = np.clip(latent_coords[:,1], 0, VIZ_SIZE_Y-1)
        emb_img[latent_coords[:,1],latent_coords[:,0]] = digit_colors
        cv2.imshow("embeddings", emb_img)
        if SAVE_FRAMES: cv2.imwrite("/tmp/vaegan-embedding-%06d.png" % batch_id, cv2.resize(emb_img,dsize=(1920,1080),interpolation=cv2.INTER_NEAREST))

        # visualize the manifold
        latents  = map(list, itertools.product(np.linspace(xmn,xmx,num=MANIFOLD_GRID), np.linspace(ymn,ymx,num=MANIFOLD_GRID)))
        manifold = np.array(encoder.generate(sess, latents))
        manifold = manifold.reshape((MANIFOLD_GRID**2,28,28), order='C')
        man_img  = np.zeros((MANIFOLD_GRID*28,MANIFOLD_GRID*28), dtype=np.float32)
        for i in range(MANIFOLD_GRID**2):
            man_img[i/MANIFOLD_GRID*28:i/MANIFOLD_GRID*28+28, i%MANIFOLD_GRID*28:i%MANIFOLD_GRID*28+28] = manifold[i]
        cv2.imshow("manifold", (man_img*255).astype(np.uint8))

        batch_id += 1

    # inspector image
    recon = encoder.generate(sess, [[xe, ye]])
    cv2.imshow("inspector", (recon*255).reshape(28,28).astype(np.uint8))

    #------------
    # USER INPUT
    #------------

    key = cv2.waitKey(10) & 0xff
    if   key == ord(' '):
        paused = not paused
    elif key == ord('r'):
        sess.run(tf.global_variables_initializer())
        sess.run(encoder.learn_rate.assign(LEARN_RATE))
        reset()
    elif key == 27:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        sys.exit(0)

