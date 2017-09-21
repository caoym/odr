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

    def __init__(self):
        self._clf = SGDClassifier(loss="modified_huber")
        self._pca = IncrementalPCA(n_components=70, whiten=True)

    def fit(self, samples_dir):
        '''
        训练
        :param samples_dir:
        :return:
        '''
        self.make_features(samples_dir)
        self.create_descriptors()
        self.cluster_words_for_labels()
        self.merge_words_for_labels()
        self.create_classifier()

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

    def get_words_count(self):
        return DB.Vocabulary.select(DB.Vocabulary.lv3).where(
            DB.Vocabulary.lv1 != -1).distinct().count()

    def get_samples(self):
        '''
        获取所有样本
        :return: {(lab,filename):[11,222,333,], ...}

        '''
        docs = {}
        for f in DB.Vocabulary.select(DB.Vocabulary.lv3, DB.Feature.label, DB.Feature.docname).join(
                DB.Feature).where(DB.Vocabulary.lv1 != -1).iterator():
            assert isinstance(f, DB.Vocabulary)
            key = (f.feature.label, f.feature.docname)
            if not docs.has_key(key):
                docs[key] = []
            docs[key].append(f.lv3)
        return docs

    def create_classifier(self):
        # DB.db.connect()
        print "start create_classifier ..."

        start = time()
        self._clf = SGDClassifier()
        print("SGDClassifier start time %s" % (start))

        query = DB.Vocabulary. \
            select(DB.Vocabulary.lv2). \
            where((DB.Vocabulary.lv1 != -1) & (DB.Vocabulary.lv2 != -1)). \
            distinct().\
            tuples().iterator()

        classes = numpy.array(map(lambda x: x[0], query))
        offset = 0
        while True:
            print ' %d SGDClassifier partial_fit %d' % (time(), offset)
            query = DB.DescriptorModel. \
                select(DB.Vocabulary.lv2, DB.DescriptorModel.lv2). \
                join(DB.Vocabulary, on=(DB.Vocabulary.feature == DB.DescriptorModel.feature)). \
                offset(offset). \
                limit(1000). \
                where((DB.Vocabulary.lv1 != -1) & (DB.Vocabulary.lv2 != -1)). \
                tuples().iterator()

            features = numpy.array(map(lambda x: [x[0]] + list(x[1]), query))
            if len(features) == 0:
                break
            X = features[:, 1:]
            Y = features[:, 0]
            self._pca.transform(X)
            self._clf.partial_fit(X, Y, classes)
            print self._clf.score(X, Y)
            offset += 1000

        print "done SGDClassifier, cost time %s" % (time() - start)

        # 使用分类器重新对所有样本分类
        print "start reclassify..."
        with DB.db.transaction():
            offset = 0
            while True:
                query = DB.DescriptorModel. \
                    select(DB.DescriptorModel.feature, DB.Vocabulary.lv1, DB.Vocabulary.lv2, DB.DescriptorModel.lv2). \
                    join(DB.Vocabulary, on=(DB.Vocabulary.feature == DB.DescriptorModel.feature)). \
                    offset(offset). \
                    limit(1000). \
                    where(DB.Vocabulary.lv1 != -1).\
                    tuples().iterator()

                features = numpy.array(map(lambda x: [x[0],x[1],x[2]] + list(x[3]), query))
                if len(features) == 0:
                    break
                X = features[:, 3:]
                feature_ids = features[:, 0]
                lv1 = features[:, 1]
                Y = features[:, 2]
                self._pca.transform(X)
                res = self._clf.predict(X)

                offset += 1000
                for i in range(0, len(res)):
                    DB.Vocabulary.update(lv3 = res[i]).where(DB.Vocabulary.feature == feature_ids[i]).execute()
        print "done create_classifier, cost time %s" % (time() - start)

    def merge_words_for_labels(self):
        '''
        合并所有分类的词汇，并重新聚类
        '''
        print "start merge_words_for_labels ..."

        start = time()

        print("merge_words_for_labels start time %s" % (start))
        print "start IncrementalPCA..."
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
        indexs = []
        for label in query:
            label = label[0]
            print "start calculate center for %s..."%(label)
            # 取每个一级分类的中心再次分类
            words_of_label = DB.Vocabulary.select(
                DB.Vocabulary.lv1
            ).distinct().where((DB.Vocabulary.label == label) & (DB.Vocabulary.lv1_core == 1)).tuples().iterator()

            for lv1 in words_of_label:
                lv1 = lv1[0]
                print "start calculate center for %s, %d..." % (label, lv1)
                lv1_query = DB.DescriptorModel. \
                    select(DB.DescriptorModel.feature, DB.DescriptorModel.lv2). \
                    join(DB.Vocabulary, on=(DB.Vocabulary.feature == DB.DescriptorModel.feature)). \
                    where((DB.Vocabulary.label == label) & (DB.Vocabulary.lv1 == lv1) & (DB.Vocabulary.lv1_core == 1)). \
                    tuples().iterator()

                features = numpy.array(map(lambda x: [x[0]] + list(x[1]), lv1_query))
                X = features[:, 1:]
                Y = features[:, 0]
                X = self._pca.transform(X)
                km = KMeans(n_clusters=1)
                km.fit(X)
                samples += [km.cluster_centers_[0]]
                indexs += [[label, lv1]]

        print "start DBSCAN..."
        samples = numpy.array(samples)
        cluster = DBSCAN(6, min_samples=1)
        res = cluster.fit_predict(samples)
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
                DB.Vocabulary.update(lv2=res[i]).where((DB.Vocabulary.label == indexs[i][0]) & (DB.Vocabulary.lv1 == indexs[i][1])).execute()

        print "done merge_words_for_labels"


    def display_words(self):
        # DB.db.connect()
        # select word_id,count(*)as count from vocabulary join feature on vocabulary.feature_id = feature.id group by word_id order by count DESC
        words = DB.Vocabulary.select(DB.Vocabulary.lv3, fn.COUNT().alias('count')) \
            .join(DB.Feature) \
            .where(DB.Vocabulary.lv1 != -1) \
            .group_by(DB.Vocabulary.lv3) \
            .order_by('count DESC')\
            .tuples().iterator()

        figure()
        line = 0
        for word in words:
            print word
            features = DB.Feature.select().\
                join(DB.Vocabulary).\
                where((DB.Vocabulary.lv3 == word[0])&(DB.Vocabulary.lv1 != -1)).\
                limit(20)
            pos = 0
            for i in range(0,min(20,len(features))):
                pos += 1
                subplot(20, 20, line * 20 + i + 1)
                axis('off')
                imshow(features[i].img)
            line += 1
            if line == 20:
                line = 0
                show()
        show()

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
