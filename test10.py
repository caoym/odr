# coding=utf-8
import os

import cPickle as pickle
import cv2
import  numpy
from matplotlib.pyplot import *
from PIL import Image
from numpy.ma import ones
from scipy import ndimage, linalg
import shutil
from time import time
from scipy.cluster.vq import *
from sklearn import preprocessing

from sklearn.decomposition import RandomizedPCA
from odr import WordCluster,DocClassifier
from train import *

from utils import AdjustPostion

from skimage import data
from skimage.filters import threshold_otsu, threshold_adaptive,gaussian_filter


from scipy.ndimage import measurements, morphology
from utils.features import getLines, getFeaturesFromImage, visualizationFeatures

from sklearn.cluster import DBSCAN

from cyvlfeat import sift
import matplotlib.pyplot as plt
from skimage import data, color, exposure
import lmdb
from skimage.feature import hog
import random
from utils.FeaturesDB import DB as FeaturesDB
from utils.FeaturesDB import SubDB as FeaturesSubDB

from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import filters
from skimage.feature import daisy

from skimage.feature import (match_descriptors, corner_peaks, corner_harris,
                             plot_matches, BRIEF)

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.cross_validation import train_test_split

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC

__author__ = 'caoym'
MaxFileSize = 5000000000
#生成特征
def genFeatures(src,dest):
    print("genFeatures %s -> %s"%(src,dest))

    from_dir = os.listdir(src)
    features = []
    count = 0

    features = FeaturesDB(dest,"features")

    for type in from_dir:
        print("genFeatures for type %s..."%type)

        type_dir = "%s/%s"%(src,type)
        if not os.path.isdir(type_dir):
            continue
        files = os.listdir(type_dir)
        db_dir = "%s/features"%dest
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
        subdb = features.newSubDB(type,True)

        pos = 0
        for f in files:
            from_file = type_dir+"/"+f
            print("processing %s..."%from_file)
            img = Image.open(from_file).convert('L')
            img = numpy.array(img,'uint8')
            res = getFeaturesFromImage(img)
            print("%s features found"%len(res))

            for i in res:
                subdb.addFeature([i,f])
                count +=1
        subdb.flush()

    print("genFeatures done. %s features found"%count)

def loadData(path,data_type, start=0,size=-1):
    dir = os.listdir(path+"/"+data_type)
    count=0
    pos = 0
    for type in dir:
        env = lmdb.open(path+"/"+data_type+"/"+type)
        with env.begin(write=False) as txn:
            i = 0
            while True:
                value = txn.get(str(i))
                if value is None:
                    break
                i+=1
                pos += 1
                if pos-1<start:
                    continue

                data = numpy.array(pickle.loads(value))
                #features[type].append(data)
                yield type,data
                count += 1
                if size>=0 and count>=size:
                    break


def getFeatures(path, start=0,size=-1):
    return loadData(path,"features",start,size)

def pca( X):
    """ 主 成分 分析： 输入： 矩阵 X ，其中 该 矩阵 中 存储 训练 数据， 每一 行为 一条 训练 数据 返回： 投影 矩阵（ 按照 维 度 的 重要性 排序）、 方差 和 均值"""
    # 获取 维 数
    num_data, dim = X.shape
    # 数据 中心 化
    mean_X = X.mean( axis= 0)
    X = X - mean_X
    if True or dim> num_data:
        # PCA- 使用 紧 致 技巧
        M = numpy.dot( X, X.T) # 协 方差 矩阵
        e, EV = linalg.eigh( M) # 特征值 和 特征 向量
        tmp = numpy.dot( X.T, EV). T   # 这 就是 紧 致 技巧
        V = tmp[::- 1]  # 由于 最后 的 特征 向量 是我 们 所需 要的， 所以 需要 将其 逆转
        S = numpy.sqrt(e)[::- 1] # 由于 特征值 是按 照 递增 顺序 排列 的， 所以 需要 将其 逆转
        for i in range( V.shape[1]):
            V[:, i] /= S
    else: # PCA- 使用 SVD 方法
        U, S, V = linalg.svd(X)
        V = V[: num_data] # 仅仅 返回 前 nun_ data 维 的 数据 才
    # 返回 投影 矩阵、 方差 和 均值
    return V, S, mean_X


#PCA降维
def genEigenvectors(path, dim=256):
    #features = getHog(path)
    hogs = FeaturesDB(path,"hogs")
    features = []
    for db in hogs.getSubDBs():
        features += list(db.getFeatures())
        break

    features = numpy.array(features)
    print("genEigenvectors %s,count=%s,dim=%s"%(path,len(features),dim))
    start = time()
    print("start time %s"%start)

    #eigenvectors, S, mean = pca(features)
    #eigenvectors=eigenvectors[:dim]
    #
    #pca = RandomizedPCA(n_components=dim)
    #pca.fit(features)
    #mean = pca.mean_
    #eigenvectors = pca.components_
    #variances = pca.explained_variance_ratio_
    variances = numpy.array([])
    mean, eigenvectors = cv2.PCACompute(features,None,maxComponents=dim)

    print("cost time %s"%(time()-start))
    print("saving eigenvectors...")

    f = open(path+"/mean",'wb')
    data = pickle.dumps(mean.tolist(),protocol=1)
    f.write(data)
    f.close()

    f = open(path+"/variances",'wb')
    data = pickle.dumps(variances.tolist(),protocol=1)
    f.write(data)
    f.close()

    env = lmdb.open(path+"/eigenvectors",map_size=MaxFileSize,max_dbs=1)
    db = env.open_db()
    with env.begin(write=True) as txn:
        txn.drop(db)
        pos = 0
        for i in eigenvectors:
            data = pickle.dumps(i.tolist(),protocol=1)
            txn.put(str(pos),data)
            pos += 1
    env.sync(True)
    env.close()
    print("genEigenvectors done")


def genHog2(path):
    print("genHog %s"%(path))
    start = time()
    dbs = {}
    features = FeaturesDB(path,"features")
    hogs = FeaturesDB(path,"hogs")
    for type in features.getSubDBs():
        subdb = hogs.newSubDB(type.name,True)
        for i in type.getFeatures():
            x = i.sum(axis=0)/255./32.
            y = i.sum(axis=1)/255./64.
            xy = x.tolist() + y.tolist()
            xy = filters.gaussian_filter(xy,2)
            i = filters.gaussian_filter(i,2)
            fd = hog(i, orientations=8, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualise=False)
            subdb.addFeature(xy.tolist()+fd.tolist())
        subdb.flush()

    print("genHog done")

def genHog(path):
    print("genHog %s"%(path))
    start = time()
    dbs = {}
    features = FeaturesDB(path,"features")
    hogs = FeaturesDB(path,"hogs")
    for type in features.getSubDBs():
        subdb = hogs.newSubDB(type.name,True)
        for i in type.getFeatures():
            #fd = hog(i[0], orientations=4, pixels_per_cell=(6, 6),cells_per_block=(2, 2), visualise=False)

            descs = daisy(numpy.array(i[0]),step=4,rings=1,histograms=8, visualize=False)

            #brief = BRIEF(patch_size=32)
            #brief.extract(numpy.array(i[0]), numpy.array([[16,16],[16,32],[16,48]]))

            '''figure()
            gray()
            axis('off')
            subplot(2,1,0)
            imshow(i[0])
            subplot(2,1,1)
            imshow(img)
            show()'''
            #subdb.addFeature(brief.descriptors.flatten())
            subdb.addFeature(descs.flatten())
        subdb.flush()

    print("genHog done")

def getHog(path,start=0,size=-1):
    return flattenDictOnce(loadData(path,"hogs",start,size));

def getEigenvectors(path):
    f = open(path+"/mean",'rb')
    data = f.read()
    mean = [pickle.loads(data)]
    f.close()

    f = open(path+"/variances",'rb')
    data = f.read()
    variances = [pickle.loads(data)]
    f.close()

    db = lmdb.open(path+"/eigenvectors")
    txn = db.begin(write=False)

    i = 0
    eigenvectors = []
    while(True):
        data = txn.get(str(i))
        i += 1
        if data is None:
            break
        data = numpy.array(pickle.loads(data))
        eigenvectors.append(data)

    return mean,eigenvectors,variances

def showFeatures(path,start,size):

    features = FeaturesDB(path,"features")
    imgs = list(list(features.getSubDBs())[0].getFeatures(0,100))

    figure()
    gray()
    size  = len(imgs)
    for i in range(0,size):
        img = numpy.array(imgs[i][0])
        subplot((size+9)/10,10,i+1)
        axis('off')
        imshow(img)

def showEigenvectors(path):
    mean,eigenvectors,variances = getEigenvectors(path)
    figure()
    gray()
    imshow(numpy.array(mean).reshape((32,64)))

    #eigenvectors
    figure()
    size = len(eigenvectors)
    for i in range(0,size):
        img = numpy.array(eigenvectors[i])
        img = img.reshape((32,64))
        subplot((size+9)/10,10,i+1)
        axis('off')
        imshow(img)

def genPcaProjects(path):

    mean,eigenvectors,variances = getEigenvectors(path)
    mean = numpy.array(mean)
    eigenvectors = numpy.array(eigenvectors)

    print("genPcaProjects ...")
    hogs = FeaturesDB(path,"hogs")
    projects = FeaturesDB(path,"projects")

    features = []
    for db in hogs.getSubDBs():
        p = projects.newSubDB(db.name,True)
        res = cv2.PCAProject(numpy.array(list(db.getFeatures())), mean[0], eigenvectors);
        for i in res:
            p.addFeature(i)
        p.flush()
    print("genPcaProjects done")

def getPcaProjects(path,start=0,size=-1):
    return loadData(path,"projects",start,size)

def showBackProjects(path,start,size):
    #projects = numpy.array(getPcaProjects(path,start,size).values()).flatten()
    types = getPcaProjects(path,start,size).values()
    projects = []
    for v in types:
        projects += v
    projects = numpy.array(projects)

    mean,eigenvectors,variances = getEigenvectors(path)

    figure()
    gray()
    features = cv2.PCABackProject(projects,numpy.array(mean[0]),numpy.array(eigenvectors))
    i=0
    size = len(features)
    for i in range(0,size):
        img = numpy.array(features[i])
        img = img.reshape((32,64))
        subplot((size+9)/10,10,i+1)
        axis('off')
        imshow(img)

def flattenDictOnce(src):
    for v in src.values():
        for i in v:
            yield i

def getFittedFeatures(path,start=0,size=-1):
    projects = flattenDictOnce(getPcaProjects(path,start,size))
    return projects;
    _,_,variances = getEigenvectors(path)
    scaler = preprocessing.MinMaxScaler()
    variances = scaler.fit_transform(numpy.array(variances[0]))
    fitProjects = projects*variances

    return fitProjects

#构建视觉单词池袋模型
def genBOW(path, count):
    data = getFittedFeatures(path,0,2000)
    #random.shuffle(data)
    start = time()
    print "start time %s"%start

    bow = cv2.BOWKMeansTrainer(count)
    data = numpy.array(data,"float32")
    center = bow.cluster(data);
    #db = DBSCAN(min_samples=2,eps=8,algorithm='ball_tree',leaf_size=100).fit(data)
    #data=None
    #center = db.components_
    print "cost time %s"%(time()-start)

    f = open(path+"/bow",'wb')
    data = pickle.dumps(center.tolist(),protocol=1)
    f.write(data)
    f.close()

def genBOWForHog(path):
    hogs = FeaturesDB(path,"projects")
    features = FeaturesDB(path,"features")
    words = FeaturesDB(path,"words")
    ori = []
    for db in hogs.getSubDBs():
        ori += list(db.getFeatures())
    ori = numpy.array(ori)
    map = numpy.arange(0,len(ori)).tolist()
    random.shuffle(map)
    data = numpy.array(ori[map])

    start = time()
    print "genBOWForHog start time %s for %s %d hogs"%(start,db.name,len(data))

    '''bow = cv2.BOWKMeansTrainer(count)
    data = numpy.array(data,"float32")
    center = bow.cluster(data);'''

    dbscan = DBSCAN(eps=0.03,algorithm='ball_tree', min_samples=3).fit(data)
    '''db = KMeans(n_clusters=500)
    db.fit_predict(numpy.array(data))'''
    print "cost time %s"%(time()-start)

    types = {}
    for i in range(0,len(dbscan.labels_)):
        type = dbscan.labels_[i]
        if not types.has_key(type):
            types[type] = []
        types[type].append(map[i])

    types= sorted(types.iteritems(),key=lambda i:len(i[1]),reverse=True)

    worddb = words.newSubDB('all',True)
    worddb.setFeature('words',types)
    components = []
    for i in dbscan.core_sample_indices_:
        components.append(map[i])
    worddb.setFeature('core_samples',components)
    worddb.flush()

def showBOW(path):
    dir = os.listdir(path+'/bow/')
    hogs = FeaturesDB(path,"hogs")
    features = FeaturesDB(path,"features")
    words = FeaturesDB(path,"words")
    types = words.getSubDB("all").getFeature('words')
    print 'words',len(types)
    figure()
    gray()
    for k,v in types:
        if k ==-1:
            continue
        print k,v;
        for i in range(0,min(400,len(v))):
            subplot(20,20,i+1)
            axis('off')
            img = features.getItemById(v[i])[0]
            imshow(img)
        show()

def genNeighbors(path):
    types = loadData(path,"hogs")

    neigh = NearestNeighbors(n_neighbors=1)
    hogs = FeaturesDB(path,"hogs")

    items = []
    types = []
    x = 0
    for type in hogs.getSubDBs():
        print(1,time())
        for i in type.getFeatures():
            types.append(int(type.name))
            items.append(i)

    print(2,time())
    neigh.fit(items)

    f = open(path+"/neighbors",'wb')
    pickle.dump(neigh,f,protocol=1)
    f.close()

def genTfIdf(path):

    print "genTfIdf ... "
    features = FeaturesDB(path,"features")
    words = FeaturesDB(path,"words").getSubDB("all").getFeature("words")
    words = dict(words)
    words.pop(-1,None)
    docs = {}
    count =0
    wordscount = len(words.keys())
    for k,v in words.iteritems():
        for i in v:
            _, doc = features.getItemById(i)
            key = (features.getSubDBById(i).name, doc)
            if not docs.has_key(key):
                docs[key]=numpy.zeros(wordscount).tolist()
            docs[key][k]+=1
            count += 1
            #print count,key,k,i

    transf = TfidfTransformer(smooth_idf=True, sublinear_tf=True,
                 use_idf=True)
    X = numpy.array(docs.values())
    tfidf = transf.fit_transform(X).toarray()

    tfidfdb = FeaturesDB(path,"tfidf")

    f = open(path+"/idf",'wb')
    pickle.dump(transf.idf_.tolist(),f,protocol=1)
    f.close()

    docname = docs.keys()
    dbs = {}
    for i in range(0,len(docname)):
        type,name = docname[i]
        if not dbs.has_key(type):
            dbs[type] = tfidfdb.newSubDB(type,True)
        dbs[type].addFeature(tfidf[i])
    for k,db in dbs.iteritems():
        db.flush()

    print "done ... "

def showTfIdf(path):

    features = FeaturesDB(path,"features")
    words = FeaturesDB(path,"words").getSubDB("all").getFeature("words")
    words = dict(words)
    words.pop(-1, None)
    f = open(path+"/idf",'rb')
    idf = pickle.load(f)
    f.close()

    test = []
    for i in words[59]:
        _, doc = features.getItemById(i)
        key = (features.getSubDBById(i).name, doc)
        test.append((key,1))
    test = dict(test)
    for i in test.keys():
        print i

    for k,v in words.iteritems():
        idf[k] = idf[k]

    id = numpy.arange(0,len(idf))
    wordsidf = numpy.column_stack((id,idf)).tolist()
    wordsidf = sorted(wordsidf, key=lambda i:i[1],reverse=False)

    figure()
    gray()
    i = 0
    for k,v in wordsidf:
        i+=1
        img = features.getItemById(words[k][0])[0]
        subplot(20,20,i)
        axis('off')
        imshow(img)
        if i == 400:
            i=0
            show()
            figure()
            gray()
    show()

def train(path):
    print "start svm train..."
    tfidfdb = FeaturesDB(path,"tfidf")
    labs = []
    data = []
    target_names = []
    for db in tfidfdb.getSubDBs():
        target_names.append(db.name)
        for i in db.getFeatures():
            data.append(i)
            labs.append(int(db.name))
    data = numpy.array(data)
    labs = numpy.array(labs)

    target_names = target_names
    n_classes = len(target_names)

    X_train, X_test, y_train, y_test = train_test_split(data, labs, test_size=0.1, random_state=42)
    ###############################################################################
    # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction
    n_components = 20
    print("Extracting the top %d eigenfaces from %d docs"
          % (n_components, X_train.shape[0]))
    t0 = time()
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
    print("done in %0.3fs" % (time() - t0))

    #eigenfaces = pca.components_.reshape((n_components, h, w))
    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))

    ###############################################################################
    # Train a SVM classification model

    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='linear',probability=True,class_weight="auto"), param_grid=param_grid)
    #'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    #clf = SVC(kernel='linear',probability=True,class_weight="auto")
    clf = clf.fit(X_train_pca, numpy.array(y_train))
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    print "score :",clf.score(X_train_pca, numpy.array(y_train))
    ###############################################################################
    # Quantitative evaluation of the model quality on the test set

    print("Predicting on the test set")
    t0 = time()
    y_pred = clf.predict(X_test_pca)

    print("done in %0.3fs" % (time() - t0))

    print(classification_report(y_test, y_pred, target_names=target_names))
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


def test3():
    src = Image.open(u"D:\\data\\samples\\物理-力学-互相作用\\1454162003.jpg$7998e6c1899978a99f3a171667888c3d").convert('L')
    src = numpy.array(src,'uint8')
    visualizationFeatures(src)

def test4():
    #genFeatures("D:/data/words","D:/data/trains")
    #showFeatures("D:/data/trains",0,100)
    #genHog("D:/data/trains")

    #genEigenvectors("D:/data/trains",70)
    #showEigenvectors("D:/data/trains")

    #genPcaProjects("D:/data/trains")
    #showBackProjects("D:/data/trains",0,100)

    #genBOWForHog("D:/data/trains",1560)
    #showBOW("D:/data/trains")
    show()

def test5():
    #genFeatures("D:/data/words","D:/data/trains")
    #showFeatures("D:/data/trains",0,100)
    #genHog("D:/data/trains")
    #genEigenvectors("D:/data/trains",70)
    #genPcaProjects("D:/data/trains")
    genBOWForHog("D:/data/trains")
    #mergeBOW("D:/data/trains")
    #showBOW("D:/data/trains")
    #genTfIdf("D:/data/trains")
    #showTfIdf("D:/data/trains")
    train("D:/data/trains")
    show()

if __name__ == '__main__':
    #test3()
    #make_features(u"D:/data/samples")
    #create_daisy_descriptors()
    #create_daisy_pca()
    #build_bow_for_labels()
    #merge_bow()
    #train_svc()

    wc = WordCluster()
    #wc.fit(u"D:/data/samples")
    #wc.create_descriptors()
    #wc.cluster_words_all()

    #wc.cluster_lv1()
    wc.cluster_lv2()
    #wc.display_words()


    dc = DocClassifier(wc)
    #dc.load()
    dc.fit()
    for k,v in dc.predict(u"D:\\data\\8.jpg").iteritems():
        print k,v