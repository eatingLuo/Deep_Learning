from PyQt4 import QtCore
from caffe_net import *
import glob
import caffe
import sklearn.metrics.pairwise
import cv2


caffemodel = './deep_model/VGG_FACE.caffemodel'
deploy_file = './deep_model/VGG_FACE_deploy.prototxt'
mean_file = None

net = Deep_net(caffemodel, deploy_file, mean_file, gpu=True)
label = ['Stranger']
db_path = './trainset'
#self.db = []

def load_db(db_path):
    global label
    label = ['Stranger']
    global db
    db = None
    if not os.path.exists(db_path):
        print('Database path is not existed!')
    folders = sorted(glob.glob(os.path.join(db_path, '*')))
    for name in folders:

        print('loading {}:'.format(name))
        label.append(os.path.basename(name))
        img_list = glob.glob(os.path.join(name, '*.jpg'))

        imgs = [cv2.imread(img) for img in img_list]
        scores, pred_labels, fea = net.classify(imgs, layer_name='fc7')

        #print('fea.shape {}'.format(fea.shape))
        fea = np.mean(fea, 0)
        if db is None:
            db = fea.copy()
        else:
            db = np.vstack((db, fea.copy()))
        print('done')
    print label


load_db(db_path)

test_path='./testset'
recognizing = True
threshold = 10
reco_name = []
if recognizing:
    img = []
    cord = []
folders = sorted(glob.glob(os.path.join(test_path, '*')))
i =0
k = 0

for name in folders:
    print('test loading {}:'.format(name))
    label.append(os.path.basename(name))
    img_list = glob.glob(os.path.join(name, '*.jpg'))
    imgs = [cv2.imread(img) for img in img_list]
    prob, pred, fea = net.classify(imgs, layer_name='fc7')

    # search from db find the closest
    #print len(db)
    dist = sklearn.metrics.pairwise.cosine_similarity(fea, db)
    #print('dist = {}'.format(dist))
    pred = np.argmax(dist, 1)
    #amin, amax = dist.min(), dist.max()
   #dist = (dist-amin)/(amax-amin)
    print('dist = {}'.format(dist))
    dist = np.max(dist, 1)
    pred = (0 if dist < threshold/100.0 else pred )
    #print('pred(after threshold) = {}'.format(pred))
    if i != pred:
        nowname = label[0]
        k = k+1
    else:
        nowname = label[i]
    reco_name.append(nowname)
    i += 1

print reco_name
print 'classify process have been done!'
print 'our accuracy of this project :'
print ((112-k)/112.0)
