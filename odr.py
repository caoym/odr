# coding=utf-8
from time import time
from scipy import signal
from scipy.cluster.vq import whiten
from skimage.morphology import skeletonize
from sklearn import preprocessing
from sklearn.cluster import DBSCAN, MiniBatchKMeans, MeanShift, KMeans
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA, RandomizedPCA, IncrementalPCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import chi2
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from peewee import *
import os
import cv2
import math
import numpy
from scipy import stats, ndimage
from scipy.ndimage import measurements, filters
from skimage.filters import threshold_adaptive
from utils import DB
from PIL import Image
from matplotlib.pyplot import *
from skimage.feature import daisy, hog
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from words import get_features_from_image


def train_svc(target_names, labs, data):
    print "start svm train..."
    ###############################################################################
    # Train a SVM classification model

    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

    gscv = GridSearchCV(SVC(), param_grid=param_grid)
    # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    # clf = SVC(kernel='linear',probability=True,class_weight="auto")
    svc = gscv.fit(data, labs)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(svc.best_estimator_)
    print "score :", svc.score(data, labs)
    return svc

def train_svc_with_test(target_names, labs, data):
    print "start svm train..."
    X_train, X_test, y_train, y_test = train_test_split(data, labs, test_size=0.1)

    ###############################################################################
    # Train a SVM classification model

    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

    #gscv = GridSearchCV(SVC(), param_grid=param_grid)
    # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    svc = SVC()
    svc.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    #print("Best estimator found by grid search:")
    #print(svc.best_estimator_)
    print "score :", svc.score(data, labs)
    ###############################################################################
    # Quantitative evaluation of the model quality on the test set

    print("Predicting on the test set")
    t0 = time()
    y_pred = svc.predict(X_test)

    print("done in %0.3fs" % (time() - t0))

    print(classification_report(y_test, y_pred, target_names=target_names))
    print(confusion_matrix(y_test, y_pred))
    return svc


class DocDescriptor(object):
    def __init__(self, word_descriptor, n_clusters=1000):
        self._n_clusters = n_clusters
        self._cluster = MiniBatchKMeans(n_clusters=n_clusters, verbose=1, max_no_improvement=None,
                                        reassignment_ratio=1.0)
        self._word_descriptor = word_descriptor

    def get_word_descriptor(self, img):
        X = get_features_from_image(img)
        words = []
        for i in X:
            words.append(self._word_descriptor.transform(i))
        return words

    def partial_fit(self, img):
        X = self.get_word_descriptor(img)
        self._cluster.partial_fit(X)

    def transform(self, img):
        X = self.get_word_descriptor(img)
        Y = self._cluster.predict(X)
        desc = [0] * self._n_clusters
        unit = 1.0 / self._n_clusters
        for i in range(0, len(Y)):
            desc[Y[i]] += unit
        return desc


class WordCluster(object):
    """
    通过聚类提取视觉词汇
    """
    def fit(self, samples_dir):
        '''
        训练
        :param samples_dir:
        :return:
        '''
        #self.make_features(samples_dir)
        #self.create_descriptors()
        #self.cluster_words_for_labels()
        self.merge_words_for_labels()
        #self.cluster_lv2()
        #self.create_classifier()

    def predict(self, img_file):
        '''
        预测文本中的文字
        :param img_file:
        :return:
        '''
        print 'start WordCluster::predict %d' % time()
        img = Image.open(img_file).convert('L')
        img = numpy.array(img, 'uint8')
        features = self.get_features_from_image(img)

        desc = []
        for i in features:
            desc.append(self.get_descriptor_lv1(i))

        # desc = self._scaler.transform(desc)
        # desc = self._pca.transform(desc)
        words = self._lv1.predict(desc)

        print 'end WordCluster::predict %d' % time()
        figure()
        gray()
        imshow(img)

        figure()
        gray()
        for i in range(0, min(400, len(features))):
            subplot(20, 20, i + 1)
            axis('off')
            imshow(features[i])

        figure()
        gray()
        for i in range(0, min(400, len(words))):
            subplot(20, 20, i + 1)
            axis('off')
            img = DB.Vocabulary \
                .select(DB.Vocabulary.feature).join(DB.Feature) \
                .where((DB.Vocabulary.lv1 == words[i]) & (DB.Vocabulary.lv2 == 0)).get().feature.img
            img = numpy.array(img)
            imshow(img)
        show()

        return words

    def predict_1(self, img_file):
        print 'start WordCluster::predict %d' % time()
        '''
        预测文本中的文字
        :param img_file:
        :return:
        '''
        img = Image.open(img_file).convert('L')
        img = numpy.array(img, 'uint8')
        features = self.get_features_from_image(img)

        desc = []
        for i in features:
            desc.append(self.get_descriptor_lv1(i))
        # 预测lv1
        lv1 = self._lv1.predict(desc)
        # 预测lv2
        lv2 = []
        for i in range(0, len(lv1)):
            lab = lv1[i]
            lv2_desc = self.get_descriptor_lv2(features[i])
            norm, pca, svc, = self._lv2[lab]
            lv2_desc = norm.transform(lv2_desc)
            lv2_desc = pca.transform(lv2_desc)
            # proba = svc.predict_proba(lv2_desc)[0]
            # lv2_lab = numpy.argmax(proba)
            # if proba[lv2_lab]>=0.0:
            lv2_lab = svc.predict(lv2_desc)[0]
            lv2.append((lab, lv2_lab))

        print 'end WordCluster::predict %d' % time()
        figure()
        gray()
        imshow(img)

        figure()
        gray()
        for i in range(0, min(400, len(features))):
            subplot(20, 20, i + 1)
            axis('off')
            imshow(features[i])

        figure()
        gray()
        for i in range(0, min(400, len(lv2))):
            subplot(20, 20, i + 1)
            axis('off')
            img = DB.Vocabulary \
                .select(DB.Vocabulary.feature).join(DB.Feature) \
                .where((DB.Vocabulary.lv1 == lv2[i][0]) & (DB.Vocabulary.lv2 == lv2[i][1])).get().feature.img
            img = numpy.array(img)
            imshow(img)
        show()

        return lv2

    def get_words_count(self):
        return DB.Vocabulary.select(DB.Vocabulary.lv2).where(
            (DB.Vocabulary.lv2 != -1) & (DB.Vocabulary.lv1 != -1)).distinct().count()

    def get_samples(self):
        '''
        获取所有样本
        :return: {(lab,filename):[11,222,333,], ...}

        '''
        docs = {}
        for f in DB.Vocabulary.select(DB.Vocabulary.lv1, DB.Vocabulary.lv2, DB.Feature.label, DB.Feature.docname).join(
                DB.Feature).where((DB.Vocabulary.lv2 != -1) & (DB.Vocabulary.lv1 != -1)).iterator():
            assert isinstance(f, DB.Vocabulary)
            key = (f.feature.label, f.feature.docname)
            if not docs.has_key(key):
                docs[key] = []
            docs[key].append(f.lv2)
        return docs

    def create_classifier(self):
        # DB.db.connect()
        clf = SGDClassifier(loss="modified_huber")
        labs_map = NameToIndex()

        with DB.db.transaction():
            offset = 0
            words_count = self.get_words_count()
            classes = numpy.arange(0, words_count)
            x_all = []
            y_all = []
            while True:
                print ' %d partial_fit %d' % (time(), offset)
                query = DB.Vocabulary \
                    .select(DB.Vocabulary.lv1, DB.Vocabulary.lv2) \
                    .join(DB.PcaModel, on=(DB.Vocabulary.feature == DB.PcaModel.feature)).order_by(
                    DB.Vocabulary.feature).offset(offset).limit(1000) \
                    .tuples().iterator()
                features = numpy.array(map(lambda x: [x[0]] + list(x[1]), query))
                offset += len(features)
                if len(features) == 0:
                    break

                Y = features[:, 0]
                X = features[:, 1:]

                labs = []
                for lab in Y:
                    labs.append(labs_map.map(lab))

                if (len(x_all) < 10000):
                    x_all = x_all + X.tolist()
                    y_all = y_all + labs
                labs = numpy.array(labs)

                # clf = LinearSVC()
                # clf = OneVsRestClassifier(SVC(probability=True, kernel='linear'))
                # clf.fit(X,labs)
                clf.partial_fit(X, labs, classes)
                print clf.score(x_all, y_all)

            DB.TrainingResult.delete().where(DB.TrainingResult.name == self.__class__.__name__ + "_clf").execute()
            DB.TrainingResult.delete().where(DB.TrainingResult.name == self.__class__.__name__ + "_labs_map").execute()

            tr = DB.TrainingResult()
            tr.name = self.__class__.__name__ + "_clf"
            tr.data = clf
            tr.save()

            tr = DB.TrainingResult()
            tr.name = self.__class__.__name__ + "_labs_map"
            tr.data = labs_map
            tr.save()

    def merge_words_for_labels(self):
        '''
        合并所有分类的词汇，并重新聚类
        '''
        print "start merge_words_for_labels ..."

        start = time()

        print("merge_words_for_labels start time %s" % (start))
        print "start IncrementalPCA..."
        self._pca = IncrementalPCA(n_components=70, whiten=True)
        offset = 0
        while True:
            print ' %d PCA partial_fit %d' % (time(), offset)
            query = DB.DescriptorModel. \
                select(DB.DescriptorModel.feature, DB.DescriptorModel.lv2). \
                join(DB.Vocabulary, on=(DB.Vocabulary.feature == DB.DescriptorModel.feature)). \
                offset(offset).\
                limit(1000).\
                where(DB.Vocabulary.lv1 != -1). \
                tuples().iterator()

            features = numpy.array(map(lambda x: [x[0]] + list(x[1]), query))
            if len(features) == 0:
                break
            X = features[:, 1:]
            Y = features[:, 0]
            self._pca.partial_fit(X)
            offset += 1000


        query = DB.Feature.select(
            DB.Feature.label
        ).distinct().tuples().iterator()

        samples = []
        for label in query:
            label = label[0]
            print "start calculate center for %s..."%(label)
            # 取每个一级分类的中心再次分类
            words_of_label = DB.Vocabulary.select(
                DB.Vocabulary.lv1
            ).distinct().where((DB.Vocabulary.label == label) & (DB.Vocabulary.lv1_core == 1)).tuples().iterator()

            for lv1 in words_of_label:
                print "start calculate center for %s, %d..." % (label, lv1)
                lv1 = lv1[0]
                lv1_query = DB.DescriptorModel. \
                    select(DB.DescriptorModel.feature, DB.DescriptorModel.lv2). \
                    join(DB.Vocabulary, on=(DB.Vocabulary.feature == DB.DescriptorModel.feature)). \
                    where((DB.Vocabulary.label == label) & (DB.Vocabulary.lv1 == lv1) & (DB.Vocabulary.lv1_core == 1)). \
                    tuples().iterator()

                features = numpy.array(map(lambda x: [x[0]] + list(x[1]), lv1_query))
                X = features[:, 1:]
                Y = features[:, 0]
                self._pca.transform(X)
                km = KMeans(n_clusters=1)
                km.fit(X)
                samples += [[label,lv1, km.cluster_centers_[0]]]

        print "start DBSCAN..."
        samples = numpy.array(samples)
        cluster = DBSCAN(6, min_samples=1)
        res = cluster.fit_predict(samples[:, 2:])
        types = {}
        for i in range(0, len(res)):
            type = res[i]
            if not types.has_key(type):
                types[type] = []
            types[type].append(i)
        print "done DBSCAN: %d words, %d core samples, %d noise" % (
             len(types.keys()), len(cluster.core_sample_indices_), len(types[-1]) if types.has_key(-1) else 0)

        print "start save..."

        with DB.db.transaction():
            for i in range(0, len(res)):
                DB.Vocabulary.update(lv2=res[i]).where((DB.Vocabulary.label == samples[i][0]) & (DB.Vocabulary.lv1 == samples[i][1])).execute()

        print "done merge_words_for_labels"


    def display_words(self):
        # DB.db.connect()
        # select word_id,count(*)as count from vocabulary join feature on vocabulary.feature_id = feature.id group by word_id order by count DESC
        words = DB.Vocabulary.select(DB.Vocabulary.lv1, DB.Vocabulary.lv2, fn.COUNT().alias('count')) \
            .join(DB.Feature) \
            .where(DB.Vocabulary.lv2 != -1) \
            .group_by(DB.Vocabulary.lv1, DB.Vocabulary.lv2) \
            .tuples().iterator()

        figure()
        for i in words:
            print i;
            features = DB.Feature.select().join(DB.Vocabulary).where(
                (DB.Vocabulary.lv1 == i[0]) & (DB.Vocabulary.lv2 == i[1])).limit(400).iterator()
            pos = 0
            for i in features:
                pos += 1
                subplot(20, 20, pos)
                axis('off')
                imshow(i.img)
            show()

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

        pass

    def cluster_all(self):
        # anova_svm = Pipeline([(,),('pca', IncrementalPCA(n_components=70)), ('svc', clf)])
        print '%d cluster_all begin' % (time())
        self._scaler = MinMaxScaler()
        # DB.db.connect()
        offset = 0
        steps = 3000
        # scale
        while True:
            print ' %d MinMaxScaler partial_fit %d' % (time(), offset)
            query = DB.DescriptorModel.select(DB.DescriptorModel.feature, DB.DescriptorModel.lv2).offset(offset).limit(
                steps).tuples().iterator()
            features = numpy.array(map(lambda x: [x[0]] + x[1].flatten().tolist(), query))
            if len(features) == 0:
                break
            offset += len(features)
            X = features[:, 1:]
            self._scaler.partial_fit(X)
        # pca
        offset = 0
        self._pca = IncrementalPCA(n_components=70, whiten=True, copy=False, )
        while True:
            print ' %d IncrementalPCA partial_fit %d' % (time(), offset)
            query = DB.DescriptorModel.select(DB.DescriptorModel.feature, DB.DescriptorModel.lv2).offset(offset).limit(
                steps).tuples().iterator()
            features = numpy.array(map(lambda x: [x[0]] + x[1].flatten().tolist(), query))
            if len(features) == 0:
                break
            offset += len(features)
            X = features[:, 1:]
            X = self._scaler.transform(X)
            self._pca.partial_fit(X)

        # KMeans
        offset = 0
        self._kmeans = MiniBatchKMeans(n_clusters=7000, verbose=1, max_no_improvement=None, reassignment_ratio=1.0)
        while True:
            print ' %d MiniBatchKMeans partial_fit %d' % (time(), offset)
            query = DB.DescriptorModel.select(DB.DescriptorModel.feature, DB.DescriptorModel.lv2).offset(offset).limit(
                30000).tuples().iterator()
            features = numpy.array(map(lambda x: [x[0]] + x[1].flatten().tolist(), query))
            if len(features) == 0:
                break
            offset += len(features)
            X = features[:, 1:]
            X = self._scaler.transform(X)
            X = self._pca.transform(X)
            self._kmeans.partial_fit(X)

        with DB.db.transaction():
            DB.Vocabulary.drop_table(fail_silently=True)
            DB.Vocabulary.create_table()

            offset = 0
            while True:
                print ' %d predict %d' % (time(), offset)
                query = DB.DescriptorModel.select(DB.DescriptorModel.feature, DB.DescriptorModel.lv2).offset(
                    offset).limit(1000).tuples().iterator()
                features = numpy.array(map(lambda x: [x[0]] + x[1].flatten().tolist(), query))
                if len(features) == 0:
                    break
                offset += len(features)
                Y = features[:, 0]
                X = features[:, 1:]
                X = self._scaler.transform(X)
                X = self._pca.transform(X)
                res = self._kmeans.predict(X)
                for i in range(0, len(res)):
                    DB.Vocabulary.insert(lv1=res[i], lv2=0, feature=Y[i]).execute()
        print '%d cluster_all end' % (time())

    def cluster_lv1(self):
        """
        对每一组样本单独聚类
        :return:
        """
        print "start cluster_lv1 ..."

        # # DB.db.connect()
        offset = 0
        limit = 3000

        with DB.db.transaction():
            DB.Vocabulary.drop_table(fail_silently=True)
            DB.Vocabulary.create_table()

        query = DB.Feature.select(DB.Feature.label).distinct().tuples().iterator()

        for label in query :

            print "start cluster_lv1 label %s ..."%label
            query = DB.DescriptorModel.\
                select(DB.DescriptorModel.feature, DB.DescriptorModel.lv2).\
                join(DB.Feature).where(DB.Feature.label == label).tuples().iterator()

            features = numpy.array(map(lambda x: [x[0]] + list(x[1]), query))
            X = features[:, 1:]
            Y = features[:, 0]

            #pca
            print "start Normalizer label %s ,size:%d..." % (label, Y.size)

            norm = preprocessing.Normalizer()
            X = norm.fit_transform(X)
            print "start RandomizedPCA label %s..." % label
            pca = RandomizedPCA(n_components=70, whiten=True)
            X = pca.fit_transform(X)

            print "start DBSCAN label %s..." % label
            cluster = DBSCAN(6, min_samples=3)
            res = cluster.fit_predict(X)

            types = {}
            for i in range(0, len(res)):
                type = res[i]
                if not types.has_key(type):
                    types[type] = []
                types[type].append(i)
            print "done DBSCAN label %s: %d words, %d core samples, %d noise" % (
                label, len(types.keys()), len(cluster.core_sample_indices_), len(types[-1]) if types.has_key(-1) else 0)

            print "start save label %s..." % label
            with DB.db.transaction():
                for i in range(0, len(res)):
                    DB.Vocabulary.insert(lv1=res[i], lv2=0, feature=Y[i]).execute()
                for i in cluster.core_sample_indices_:
                    DB.Vocabulary.update(lv1_core=True).where(DB.Vocabulary.feature == Y[i]).execute()

            print "done cluster_lv1 label %s..." % label

        print "done cluster_lv1"
        return cluster

    def cluster_lv2(self):
        print "start cluster_lv2 ..."
        # DB.db.connect()
        clusters = {}
        word_count = 0
        with DB.db.transaction():
            maxid = DB.Vocabulary.select(fn.MAX(DB.Vocabulary.lv1).alias('max')).get().max
            for lv1_id in range(0, maxid + 1):

                count = DB.Vocabulary.select(fn.COUNT().alias('count')).where(DB.Vocabulary.lv1 == lv1_id).get().count
                print "begin cluster_lv2 %d, %d" % (lv1_id, count)
                cluster = DBSCAN(6, min_samples=3)
                # cluster = MeanShift(bandwidth=0.79, cluster_all=False, min_bin_freq=3)

                query = DB.DescriptorModel. \
                    select(DB.DescriptorModel.feature, DB.DescriptorModel.lv2). \
                    join(DB.Vocabulary, on=(DB.Vocabulary.feature == DB.DescriptorModel.feature)). \
                    where(DB.Vocabulary.lv1 == lv1_id). \
                    tuples().iterator()

                features = numpy.array(map(lambda x: [x[0]] + x[1].flatten().tolist(), query))
                if len(features) == 0:
                    continue
                X = features[:, 1:]
                Y = features[:, 0]

                norm = preprocessing.Normalizer()
                X = norm.fit_transform(X)
                pca = RandomizedPCA(n_components=70, whiten=True)
                X = pca.fit_transform(X)
                try:
                    cluster.fit(X)
                except Exception, e:
                    print "** ERROR **" + e.message
                    DB.Vocabulary.update(lv2=-1).where(DB.Vocabulary.lv1 == lv1_id).execute()
                    continue
                trainX = X[cluster.labels_ != -1]
                trainY = cluster.labels_[cluster.labels_ != -1]
                if numpy.unique(trainY).size == 1:
                    continue
                try:
                    svc = train_svc(None, trainY, trainX)
                except Exception, e:
                    print "** ERROR **" + e.message
                    DB.Vocabulary.update(lv2=-1).where(DB.Vocabulary.lv1 == lv1_id).execute()
                    continue

                types = {}
                for lab in range(0, len(cluster.labels_)):
                    type = cluster.labels_[lab]
                    if not types.has_key(type):
                        types[type] = []
                    types[type].append(lab)
                print "end cluster_lv2 %d words, %d core samples, %d noise" % (
                len(types.keys()), len(cluster.core_sample_indices_), len(types[-1]) if types.has_key(-1) else 0)
                # print "end cluster_lv2 %d words, %d core centers, %d noise"%(len(types.keys()),len(cluster.cluster_centers_),  len(types[-1]) if types.has_key(-1) else 0 )

                word_count += len(types.keys())
                # figure()
                # line = 0
                # for k,v in types.iteritems():
                #     if len(v)<2:
                #         continue
                #     if k ==-1:
                #         continue
                #     print k,v;
                #     for i in range(0,min(20,len(v))):
                #         subplot(20,20,line*20+i+1)
                #         axis('off')
                #         f = DB.Feature(DB.Feature.img).get(DB.Feature.id == Y[v[i]])
                #         imshow(f.img)
                #     line += 1
                #     if line == 20:
                #         line = 0
                #         show()
                # show()

                for id in range(0, len(cluster.labels_)):
                    type = cluster.labels_[id]
                    DB.Vocabulary.update(lv2=type).where(DB.Vocabulary.feature == Y[id]).execute()
                clusters[lv1_id] = [norm, pca, svc]

        self._lv2 = clusters
        print "done cluster_lv2"

    def cluster_words_all(self):
        '''
        对所有样本进行聚类
        '''

        print "start cluster_words_all ..."
        offset = 0
        limit = 300
        cluster = MiniBatchKMeans(n_clusters=100, verbose=1)
        while True:
            print ' %d partial_fit %d' % (time(), offset)

            query = DB.PcaModel.select(DB.PcaModel.feature, DB.PcaModel.pca) \
                .offset(offset).limit(limit).tuples().iterator()

            features = numpy.array(map(lambda x: [x[0]] + list(x[1]), query))
            if len(features) == 0:
                break
            offset += len(features)
            X = features[:, 1:]
            cluster.partial_fit(X)

        # DB.db.connect()
        with DB.db.transaction():
            DB.Vocabulary.drop_table(fail_silently=True)
            DB.Vocabulary.create_table()
            DB.Words.drop_table(fail_silently=True)
            DB.Words.create_table()

            offset = 0
            while True:
                query = DB.PcaModel.select(DB.PcaModel.feature, DB.PcaModel.pca).offset(offset).limit(
                    1000).tuples().iterator()
                features = numpy.array(map(lambda x: [x[0]] + list(x[1]), query))
                if len(features) == 0:
                    break
                offset += len(features)
                X = features[:, 1:]
                Y = features[:, 0]
                res = cluster.predict(X)

                for i in range(0, len(res)):
                    DB.Words.insert(id=res[i]).upsert().execute()
                    DB.Vocabulary.insert(word=res[i], feature=Y[i]).execute()

                DB.TrainingResult.delete().where(DB.TrainingResult.name == self.__class__.__name__ + "_clf").execute()

                tr = DB.TrainingResult()
                tr.name = self.__class__.__name__ + "_clf"
                tr.data = cluster
                tr.save()

        # print "%d words, %d core samples, %d noise"%(len(types.keys()),len(res.core_sample_indices_), len(types[-1]) )

        print "done cluster_words_all"
        # self.display_words()
        return cluster

    def cluster_words_for_labels(self):
        '''
        对每类文本进行聚类，提取词汇
        :return:
        '''
        # DB.db.connect()
        with DB.db.transaction():
            DB.Vocabulary.drop_table(fail_silently=True)
            DB.Vocabulary.create_table()
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
                self.cluster_words_for_label(label[0])

    def cluster_words_for_label(self, label):
        '''
        每个分类独立聚类
        '''
        print "start cluster_words_for_label %s ..." % label
        start = time()
        query = DB.DescriptorModel. \
            select(DB.DescriptorModel.feature, DB.DescriptorModel.lv2). \
            join(DB.Feature).where(DB.Feature.label == label).tuples().iterator()

        features = numpy.array(map(lambda x: [x[0]] + list(x[1]), query))
        X = features[:, 1:]
        Y = features[:, 0]

        print("cluster_words_for_label %s start time %s,%d features" % (label, start, len(features)))
        # pca
        print "start Normalizer label %s ,size:%d..." % (label, Y.size)

        norm = preprocessing.Normalizer()
        X = norm.fit_transform(X)
        print "start RandomizedPCA label %s..." % label
        pca = RandomizedPCA(n_components=70, whiten=True)
        X = pca.fit_transform(X)

        print "start DBSCAN label %s..." % label
        cluster = DBSCAN(6, min_samples=3)
        res = cluster.fit_predict(X)

        types = {}
        for i in range(0, len(cluster.labels_)):
            type = cluster.labels_[i]
            if not types.has_key(type):
                types[type] = []
            types[type].append(i)
        print "done DBSCAN label %s: %d words, %d core samples, %d noise" % (
            label, len(types.keys()), len(cluster.core_sample_indices_), len(types[-1]) if types.has_key(-1) else 0)


        types = sorted(types.iteritems(), key=lambda i: len(i[1]), reverse=True)


        print "start save label %s..." % label
        with DB.db.transaction():
            for i in range(0, len(res)):
                DB.Vocabulary.insert(lv1=res[i], lv2=0, feature=Y[i], lv1_core=False, label=label).execute()
            for i in cluster.core_sample_indices_:
                DB.Vocabulary.update(lv1_core=True).where(DB.Vocabulary.feature == Y[i]).execute()

        # figure()
        # line = 0
        # for k,v in types:
        #     if k ==-1:
        #         continue
        #     print k,v
        #     for i in range(0,min(20,len(v))):
        #         subplot(20,20,line*20+i+1)
        #         axis('off')
        #         f = DB.Feature(DB.Feature.img).get(DB.Feature.id == Y[v[i]])
        #         imshow(f.img)
        #     line += 1
        #     if line == 20:
        #         line = 0
        #         show()
        # show()

        # for k,v in types:
        #     #if k ==-1:
        #     #    continue
        #     print k,v
        #     for i in range(0,min(400,len(v))):
        #         subplot(20,20,i+1)
        #         axis('off')
        #         f = DB.Feature(DB.Feature.img).get(DB.Feature.id == Y[v[i]])
        #         imshow(f.img)
        #     show()
        print "done cluster_words_for_label, cost time %s" % (time() - start)

    def create_descriptors_pca(self, dim=90):
        '''
        计算描述子pca
        :param dim:
        :return:
        '''
        print("start create_descriptors_pca ...")
        query = DB.DescriptorModel.select(DB.DescriptorModel.id, DB.DescriptorModel.descriptor).tuples().iterator()
        features = numpy.array(map(lambda x: [x[0]] + list(x[1]), query))
        print("create_descriptors_pca,count=%d,dim=%d" % (len(features), dim))
        start = time()
        print("build eigenvectors start time %s" % start)

        mean, eigenvectors = cv2.PCACompute(features[:, 1:], None, maxComponents=dim)
        fitted = cv2.PCAProject(features[:, 1:], mean, eigenvectors)
        # pca = PCA(n_components=dim)
        # fitted = pca.fit_transform(features[:,1:])
        print("build eigenvectors cost time %s" % (time() - start))
        print("saving data ...")

        # scaler = preprocessing.MinMaxScaler()
        # pca = scaler.fit_transform(pca)
        # DB.db.connect()
        with DB.db.transaction():
            DB.PcaModel.drop_table(fail_silently=True)
            DB.PcaModel.create_table()

            # res = DB.TrainingResult()
            # res.name = "daisy_pca"
            # res.data = pca
            # res.save()

            for i in range(0, len(fitted)):
                model = DB.PcaModel()
                model.pca = fitted[i]
                model.feature = features[i][0]
                model.save()

            DB.TrainingResult.delete().where(DB.TrainingResult.name == "pca_mean").execute()
            DB.TrainingResult.delete().where(DB.TrainingResult.name == "pca_eigenvectors").execute()
            tr = DB.TrainingResult()
            tr.name = "pca_mean"
            tr.data = mean
            tr.save()

            tr = DB.TrainingResult()
            tr.name = "pca_eigenvectors"
            tr.data = eigenvectors
            tr.save()

        print("create_descriptors_pca done")

    def get_descriptor_lvX(self, img):
        ori = img
        # img = cv2.bitwise_not(numpy.array(img))
        # img = threshold_adaptive(numpy.array(img), 40)
        # img = cv2.bitwise_not(img*255.)
        img = skeletonize(numpy.array(img) / 255.) * 255.
        '''figure()
        gray()
        subplot(221)
        imshow(ori)
        subplot(222)
        imshow(img)
        show()'''
        # e = stats.entropy(img.flatten())
        # if math.isnan(e) or math.isinf(e):
        #    return 0
        # else:
        #    return e
        descs = hog(numpy.array(img), orientations=4, pixels_per_cell=(10, 10), cells_per_block=(3, 3), visualise=False)
        '''figure()
        gray()
        imshow(img)
        figure()
        imshow(hpgimg)
        show()'''
        return descs

    def get_descriptor_lv2(self, img):


        img = cv2.Canny(numpy.uint8(img), 50, 200)
        # img = numpy.array(img.resize((48,16),Image.BILINEAR) )
        # img = cv2.bitwise_not(numpy.array(img))
        # img = threshold_adaptive(numpy.array(img), 40)
        descs = hog(img, orientations=6, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualise=False)
        # descs = hog(img, orientations=6, pixels_per_cell=(4, 4),cells_per_block=(3, 3),visualise=False)
        # descs2 = hog(img, orientations=9, pixels_per_cell=(16, 16),cells_per_block=(3, 3))
        # descs3 = hog(f.ori, orientations=9, pixels_per_cell=(32, 32),cells_per_block=(1, 2))
        # descs = descs2.flatten().tolist() + descs.flatten().tolist()
        '''figure()
        gray()
        imshow(img)
        figure()
        imshow(hpgimg)
        show()'''
        return descs

    def get_descriptor_lv1(self, img):
        # 取每一列的最高点和最低点组合成特征描述符

        img = skeletonize(numpy.array(img) / 255.)

        #top = []  # 上边曲线
        #bottom = []  # 下边曲线
        jumps = []  # 每一列的跳跃次数
        h, w = img.shape
        for i in range(0, w):
            y = numpy.where(img[:, i] != 0)[0]
            if len(y) == 0:
                #top.append(h)
                #bottom.append(h)
                jumps.append(0)
            else:
                #top.append(y.min() + 1)
                #bottom.append(h - y.max() - 1)
                jump = 0
                last = 0
                for i in y:
                    if last != i:
                        jump += 1
                    last = i + 1
                if last != h:
                    jump += 1
                jumps.append(jump)

        #top = numpy.array(top, 'float') / h
        #bottom = numpy.array(bottom, 'float') / h
        th = (h - img.sum(axis=0)).astype('float') / h #每列有效像素的比例
        jumps = numpy.array(jumps, 'float') * 2. / h

        #top = filters.gaussian_filter(top, 2)
        #bottom = filters.gaussian_filter(bottom, 2)
        th = filters.gaussian_filter(th, 2)
        jumps = filters.gaussian_filter(jumps, 2)

        # figure()
        # gray()
        # subplot(221)
        # title("Original")
        # imshow(ori)
        # subplot(222)
        # title("Skeletonize")
        # imshow(img)
        # subplot(223)
        # title("White count")
        # plot(th)
        # subplot(224)
        # title("Jumps")
        # plot(jumps)
        # show()
        return th.tolist() + jumps.tolist()

    def get_descriptor2(self, img):
        img = img.convert('L')
        descs = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3))
        # descs2 = hog(img, orientations=9, pixels_per_cell=(16, 16),cells_per_block=(3, 3))
        # descs3 = hog(f.ori, orientations=9, pixels_per_cell=(32, 32),cells_per_block=(1, 2))
        # descs = descs2.flatten().tolist() + descs.flatten().tolist()
        # descs = daisy(f.ori,step=6,rings=2,histograms=8, visualize=False)
        # brief = BRIEF(patch_size=32)
        # brief.extract(numpy.array(i[0]), numpy.array([[16,16],[16,32],[16,48]]))
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
        return descs

    def create_descriptors(self):
        '''
        生成特征描述子
        :return:
        '''
        print("start create_descriptors")
        start = time()
        dbs = {}
        count = 0
        with DB.db.transaction():
            DB.DescriptorModel.drop_table(fail_silently=True)
            DB.DescriptorModel.create_table()
            lv1 = []
            for f in DB.Feature.select(DB.Feature.id, DB.Feature.ori).iterator():
                assert isinstance(f, DB.Feature)

                model = DB.DescriptorModel()
                model.lv1 = self.get_descriptor_lv1(f.ori)
                model.lv2 = self.get_descriptor_lv2(f.ori)
                model.feature = f.id
                model.save()
                count += 1
                if count % 100 == 0:
                    print "did %d features" % count

        print "did %d features" % count
        print("create_descriptors done")

    def make_features(self, samples_dir):
        '''
        从图片中提取特征
        :param samples_dir:
        :return:
        '''
        print("WordCluster::make_features %s" % (samples_dir))
        with DB.db.transaction():
            DB.Feature.drop_table(fail_silently=True)
            DB.Feature.create_table()
            DB.TrainingResult.create_table(fail_silently=True)
            from_dir = os.listdir(samples_dir)
            features = []
            count = 0

            for type in from_dir:
                print("get_features_from_image for type %s..." % type)
                type_dir = "%s/%s" % (samples_dir, type)
                if not os.path.isdir(type_dir):
                    continue
                files = os.listdir(type_dir)

                for f in files:
                    if f[-4:] != ".jpg":
                        continue
                    from_file = type_dir + "/" + f
                    print("processing %s..." % from_file)
                    img = Image.open(from_file).convert('L')
                    img = numpy.array(img, 'uint8')
                    res = get_features_from_image(img)
                    print("%s features found" % len(res))

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
            print("WordCluster::make_features done. %s features found" % count)


class NameToIndex:
    def __init__(self):
        self.buf = {}
        self.names = []

    def map(self, name):
        if self.buf.has_key(name):
            return self.buf[name]
        else:
            id = len(self.names)
            self.buf[name] = id
            self.names.append(name)
            return id

    def name(self, id):
        return self.names[id]


class DocClassifier(object):
    """
    文档分类
    """

    def __init__(self, word_cluster):
        self._word_cluster = word_cluster

    def fit(self):

        wordids_map = NameToIndex()
        labs_map = NameToIndex()

        wordscount = self._word_cluster.get_words_count()
        print "start compute_tfidf ..."
        # 计算文档的词袋模型
        docs = self._word_cluster.get_samples()
        count = 0
        bow = []
        labs = []

        for k, v in docs.iteritems():
            vec = numpy.zeros(wordscount).tolist()
            for i in v:
                vec[wordids_map.map(i)] += 1
            bow.append(vec)
            labs.append(labs_map.map(k[0]))

        labs = numpy.array(labs)

        tfidf = TfidfTransformer(smooth_idf=True, sublinear_tf=True, use_idf=True)
        datas = numpy.array(tfidf.fit_transform(bow).toarray())

        print "compute_tfidf done"
        pca = RandomizedPCA(n_components=20, whiten=True).fit(datas)
        svc = train_svc_with_test(numpy.array(labs_map.names), labs, pca.transform(datas))

        self._tfidf = tfidf
        self._svc = svc
        self._labs_map = labs_map
        self._wordids_map = wordids_map
        self._pca = pca

    def predict(self, img_file):
        doc_words = self._word_cluster.predict(img_file)
        vec = numpy.zeros(self._word_cluster.get_words_count()).tolist()
        for i in doc_words:
            if i != -1:
                vec[self._wordids_map.map(i)] += 1

        tfidf = numpy.array(self._tfidf.fit_transform(vec).toarray())

        tfidf = self._pca.transform(tfidf)
        res = {}
        i = 0
        for score in self._svc.predict_proba(tfidf)[0]:
            res[self._labs_map.names[i]] = score
            i += 1
        return res
