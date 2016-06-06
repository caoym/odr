# coding=utf-8
import os
from PIL import Image
import cv2
from matplotlib.pyplot import imsave
import numpy
import sqlite3
from sklearn import preprocessing
from sklearn.cluster import DBSCAN, MiniBatchKMeans,Birch,KMeans
from sklearn.decomposition import RandomizedPCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from utils.features import getFeaturesFromImage
from matplotlib.pyplot import *
from time import time
from StringIO import StringIO
from skimage.feature import daisy, hog
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
import hdbscan
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cluster import MeanShift

from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy import stats
__author__ = 'caoym'
import utils.DB as DB

def make_features(src):
    print("make_features %s"%(src))

    DB.db.connect()
    with DB.db.transaction():
        DB.Feature.drop_table(fail_silently=True)
        DB.Feature.create_table()
        DB.TrainingResult.create_table(fail_silently=True)
        from_dir = os.listdir(src)
        features = []
        count = 0

        for type in from_dir:
            print("genFeatures for type %s..."%type)

            type_dir = "%s/%s"%(src,type)
            if not os.path.isdir(type_dir):
                continue
            files = os.listdir(type_dir)

            for f in files:
                from_file = type_dir+"/"+f
                print("processing %s..."%from_file)
                img = Image.open(from_file).convert('L')
                img = numpy.array(img,'uint8')
                res = getFeaturesFromImage(img)
                print("%s features found"%len(res))

                for i in res:
                    count += 1
                    data = numpy.array(i).tolist()
                    mode = DB.Feature()
                    mode.ori = data
                    mode.img = i
                    mode.label = type
                    mode.docname = f
                    mode.entropy = stats.entropy(numpy.array(data).flatten())
                    mode.save()
        print("make_features done. %s features found"%count)

def create_daisy_descriptors():
    print("start create_daisy_descriptors")
    start = time()
    dbs = {}
    count = 0
    with DB.db.transaction():
        DB.DescriptorModel.drop_table(fail_silently=True)
        DB.DescriptorModel.create_table()
        for f in DB.Feature.select(DB.Feature.id,DB.Feature.ori).iterator():
            assert isinstance(f, DB.Feature)
            descs = hog(f.ori, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(3, 3))
            descs2 = hog(f.ori, orientations=9, pixels_per_cell=(16, 16),cells_per_block=(3, 3))
            #descs3 = hog(f.ori, orientations=9, pixels_per_cell=(32, 32),cells_per_block=(1, 2))
            descs = descs2.flatten().tolist() + descs.flatten().tolist()
            #descs = daisy(f.ori,step=6,rings=2,histograms=8, visualize=False)
            #brief = BRIEF(patch_size=32)
            #brief.extract(numpy.array(i[0]), numpy.array([[16,16],[16,32],[16,48]]))
            '''figure()
            gray()
            axis('off')
            subplot(221)
            imshow(f.ori)
            subplot(222)
            imshow(img3)
            subplot(224)
            imshow(img)
            subplot(223)
            imshow(img2)
            show()'''

            model = DB.DescriptorModel()
            model.descriptor = descs
            model.feature = f.id
            model.save()
            count += 1
            if count %100 == 0:
                print "did %d features"%count

    print "did %d features"%count
    print("create_daisy_descriptors done")

def create_daisy_pca(dim = 90):

    print("start create_daisy_pca ...")
    query = DB.DescriptorModel.select(DB.DescriptorModel.id,DB.DescriptorModel.descriptor).tuples().iterator()
    features = numpy.array(map(lambda x:[x[0]]+list(x[1]),query))
    print("create_daisy_apca,count=%d,dim=%d"%(len(features),dim))
    start = time()
    print("build eigenvectors start time %s"%start)

    mean, eigenvectors = cv2.PCACompute(features[:,1:],None,maxComponents=dim)
    fitted = cv2.PCAProject(features[:,1:],mean, eigenvectors)
    #pca = PCA(n_components=dim)
    #fitted = pca.fit_transform(features[:,1:])
    print("build eigenvectors cost time %s"%(time()-start))
    print("saving data ...")

    #scaler = preprocessing.MinMaxScaler()
    #pca = scaler.fit_transform(pca)

    with DB.db.transaction():
        DB.PcaModel.drop_table(fail_silently=True)
        DB.PcaModel.create_table()

        #res = DB.TrainingResult()
        #res.name = "daisy_pca"
        #res.data = pca
        #res.save()

        for i in range(0,len(fitted)):
            model = DB.PcaModel()
            model.pca = fitted[i]
            model.feature = features[i][0]
            model.save()

    print("create_daisy_pca done")

def merge_bow():

    '''
    '''
    print "start merge_bow ..."
    query = DB.PcaModel.select(DB.PcaModel.feature,DB.PcaModel.pca)\
        .join(DB.SubVocabulary,on =(DB.PcaModel.feature == DB.SubVocabulary.feature))\
        .where(DB.SubVocabulary.word != -1).tuples().iterator()

    features = numpy.array(map(lambda x:[x[0]]+list(x[1]),query))

    print "%d features"%(len(features))
    start = time()
    print "start time %s "%(start)

    index = features[:,0]
    data = features[:,1:]

    prepro = preprocessing.Normalizer()
    data = prepro.fit_transform(data)
    features = None

    '''bow = cv2.BOWKMeansTrainer(50000)
    data = numpy.array(data,"float32")
    center = bow.cluster(data);'''
    cluster = DBSCAN(0.52,algorithm='ball_tree', min_samples=2,leaf_size=3000)
    #cluster = Birch(threshold=0.028, n_clusters=None,copy=False)
    #cluster = MeanShift();
    #
    #cluster = MiniBatchKMeans(init='k-means++', n_clusters=50000,random_state=0,batch_size=50000,reassignment_ratio=0,verbose=1,max_iter=100)
    res = cluster.fit_predict(data)

    #cluster = KMeans(n_clusters=50000);
    print "cost time %s"%(time()-start)

    types = {}
    for i in range(0,len(res.labels_)):
        type = res.labels_[i]
        if not types.has_key(type):
            types[type] = []
        types[type].append(i)

    #print "%d words, %d core samples, %d noise"%(len(types.keys()),len(res.core_sample_indices_), len(types[-1]) )
    print "%d words"%len(types.keys())
    types = sorted(types.iteritems(),key=lambda i:len(i[1]),reverse=True)

    '''figure()
    line = 0
    for k,v in types:
        if k ==-1:
            continue
        print k,v;
        for i in range(0,min(20,len(v))):
            subplot(20,20,line*20+i+1)
            axis('off')
            f = DB.Feature(DB.Feature.img).get(DB.Feature.id == index[v[i]])
            imshow(f.img)
        line += 1
        if line == 20:
            line = 0
            show()
    show()'''

    '''for k,v in types:
        #if k ==-1:
        #    continue
        print k,v;
        for i in range(0,min(400,len(v))):
            subplot(20,20,i+1)
            axis('off')
            f = DB.Feature(DB.Feature.img).get(DB.Feature.id == index[v[i]])
            imshow(f.img)
        show()'''

    with DB.db.transaction():
        DB.Vocabulary.drop_table(fail_silently=True)
        DB.Vocabulary.create_table()
        DB.Words.drop_table(fail_silently=True)
        DB.Words.create_table()
        for k,v in types:
            if k == -1:
                continue
            word = DB.Words()
            word.chi = 0
            word.idf = 0
            word.ignore = False
            word.save(force_insert=True)
            for w in v:
                DB.Vocabulary.insert(word = word, feature = index[w] ).execute()

        #计算词重要性
        target_names, labs,datas, wordids = get_doc_vector()
        chi2_scores,_ = chi2(datas,labs)
        idf = TfidfTransformer(smooth_idf=True, sublinear_tf=True,use_idf=True)
        idf.fit(datas)

        for i in range(0,len(chi2_scores)):
            word = DB.Words()
            word.id = wordids[i]
            word.chi = chi2_scores[i]
            word.idf = idf.idf_[i]
            word.save()
            #DB.Words.update(chi = chi2_scores[i], idf = idf.idf_[i]).where(DB.Words.id ==wordids[i]).execute()

    print "done build_bow"

def build_bow_for_label(label):
    '''
    每个分类独立计算bow
    '''
    print "start build_bow_for_label %s ..."%label
    dim = 90
    query = DB.DescriptorModel.select(
        DB.DescriptorModel.id,DB.DescriptorModel.descriptor
    ).join(
        DB.Feature
    ).where((DB.Feature.ignore == 0) & (DB.Feature.label == label)).tuples().iterator()

    features = numpy.array(map(lambda x:[x[0]]+list(x[1]),query))
    start = time()
    print("build_bow_for_type %s start time %s,%d features"%(label,start,len(features)))

    pca = PCA(n_components=dim)
    fitted = pca.fit_transform(features[:,1:])
    print "PCA cost time %s"%(time()-start)
    start = time()
    index = features[:,0]
    data = fitted

    prepro = preprocessing.Normalizer()
    data = prepro.fit_transform(data)
    features = None

    cluster = DBSCAN(0.55, algorithm='kd_tree', min_samples=2,leaf_size=300)
    res = cluster.fit(data)

    print "cluster cost time %s"%(time()-start)

    types = {}
    for i in range(0,len(res.labels_)):
        type = res.labels_[i]
        if not types.has_key(type):
            types[type] = []
        types[type].append(i)

    print "%d words, %d core samples, %d noise"%(len(types.keys()),len(res.core_sample_indices_), len(types[-1]) )
    types = sorted(types.iteritems(),key=lambda i:len(i[1]),reverse=True)

    '''figure()
    line = 0
    for k,v in types:
        if k ==-1:
            continue
        print k,v;
        for i in range(0,min(20,len(v))):
            subplot(20,20,line*20+i+1)
            axis('off')
            f = DB.Feature(DB.Feature.img).get(DB.Feature.id == index[v[i]])
            imshow(f.img)
        line += 1
        if line == 20:
            line = 0
            show()
    show()'''
    '''
    for k,v in types:
        #if k ==-1:
        #    continue
        print k,v;
        for i in range(0,min(400,len(v))):
            subplot(20,20,i+1)
            axis('off')
            f = DB.Feature(DB.Feature.img).get(DB.Feature.id == index[v[i]])
            imshow(f.img)
        show()'''

    for k,v in types:
        if k ==-1:
            continue
        words = DB.SubWords()
        words.ignore=False
        words.label=label
        words.save()
        for w in v:
            DB.SubVocabulary.insert(word=words,feature=index[w] ).execute()

    print "done build_bow_for_type %s"%label

def build_bow_for_labels():

    with DB.db.transaction():
        DB.SubWords.drop_table(fail_silently=True)
        DB.SubVocabulary.drop_table(fail_silently=True)
        DB.SubVocabulary.create_table()
        DB.SubWords.create_table()
        '''query = DB.Feature.select(DB.Feature.id,DB.Feature.ori).distinct().iterator()
        i = 0
        for f in query:
            f.entropy = stats.entropy(numpy.array(f.ori).flatten())
            f.save()
            i += 1
            if i == 1000:
                break
        return'''
        query = DB.Feature.select(
            DB.Feature.label
        ).distinct().tuples().iterator()
        for label in query:
            build_bow_for_label(label[0])


class NameToIndex:
    def __init__(self):
        self.buf = {}
        self.names = []

    def map(self,name):
        if self.buf.has_key(name):
            return self.buf[name]
        else:
            id = len(self.names)
            self.buf[name] = id
            self.names.append(name)
            return id


def get_doc_vector():
    docs = {}
    count =0
    wordscount = DB.Vocabulary.select(DB.Vocabulary.word).join(DB.Words).where((DB.Vocabulary.word != -1) & (DB.Words.ignore == 0)).distinct().count()
    wordid_map = NameToIndex()
    for f in DB.Vocabulary.select(DB.Vocabulary.word,DB.Feature.label,DB.Feature.docname).join(DB.Feature).switch(DB.Vocabulary).join(DB.Words).where((DB.Vocabulary.word != -1) & (DB.Words.ignore == 0)).iterator():
        assert isinstance(f,DB.Vocabulary)
        key = (f.feature.label, f.feature.docname)
        if not docs.has_key(key):
            docs[key]=numpy.zeros(wordscount).tolist()
        docs[key][wordid_map.map(f.word_id)]+=1
        count += 1
            #print count,key,k,i
    labs = []
    datas = []

    for k,v in docs.iteritems():
        datas.append(v)
        labs.append(k[0])

    target_names = NameToIndex()

    for i in range(0,len(labs)):
        labs[i] = target_names.map(labs[i])

    return numpy.array(target_names.names), numpy.array(labs),numpy.array(datas),numpy.array(wordid_map.names)

def compute_tfidf():
    print "start compute_tfidf ..."
    target_names, labs, datas,wordids = get_doc_vector()

    transf = TfidfTransformer(smooth_idf=True, sublinear_tf=True,use_idf=True)
    datas = transf.fit_transform(datas).toarray()
    print "compute_tfidf done"
    return target_names, labs, numpy.array(datas)

def train_svc():
    print "start svm train..."
    target_names,labs,data = compute_tfidf()
    n_classes = len(target_names)
    X_train, X_test, y_train, y_test = train_test_split(data, labs, test_size=0.2, random_state=42)
    ###############################################################################
    # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction
    n_components = 20
    print("Extracting the top %d eigenfaces from %d docs"
          % (n_components, X_train.shape[0]))

    '''ch2 = SelectKBest(chi2, k=100)
    ch2.fit(X_train, y_train)
    X_train_pca = ch2.transform(X_train)
    X_test_pca = ch2.transform(X_test)'''


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
    clf = GridSearchCV(SVC(kernel='rbf',probability=True,class_weight="auto"), param_grid=param_grid)
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


def predict(from_file):
    print("start predict image %s"%from_file)
    #提取特征
    img = Image.open(from_file).convert('L')
    img = numpy.array(img,'uint8')
    res = getFeaturesFromImage(img)
    print("%s features found"%len(res))

    #特征 -> 词汇
    #pca


    #词汇 -> tfdif
    #


    #tfdif -> svn 预测

    return 0;
