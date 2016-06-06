# coding=utf-8
from odr import WordCluster, DocClassifier
import cPickle as pickle

__author__ = 'caoym'

def save(var, name):
    file = open(name, 'wb')
    pickle.dump(var,file,protocol=1)
    file.close()

def load(name):
    file = open(name, 'rb')
    var = pickle.load(file)
    file.close()
    return var

if __name__ == '__main__':

    #wc = WordCluster()
    wc = load('WordCluster.dump')
    #wc.fit(u"D:/data/samples")
    #wc.create_descriptors()
    #wc.cluster_words_all()
    wc.cluster_all()
    #wc.cluster_lv2()
    #save(wc,'WordCluster.dump')
    #
    save(wc,'WordCluster.dump')
    wc.display_words()

    dc = DocClassifier(wc)
    #dc = load('DocClassifier.dump.1')
    dc.fit()
    save(dc,'DocClassifier.dump.1')

    for k,v in dc.predict(u"D:\\data\\3.jpg").iteritems():
        print k,v