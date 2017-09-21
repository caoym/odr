# coding=utf-8

import sys
import getopt
import numpy
from odr import WordCluster, DocClassifier
import cPickle as pickle
from PIL import Image
from words import get_features_from_image, adjust_slope, adjust_size, visualization_features

def save(var, name):
    file = open(name, 'wb')
    pickle.dump(var, file, protocol=1)
    file.close()


def load(name):
    file = open(name, 'rb')
    var = pickle.load(file)
    file.close()
    return var


def fit(input_dir, output_dir="/Users/caoyangmin/Documents/"):

    """
    训练
    :param input_dir: 样板所在所在目录
    :param output_dir: 训练结果存放路径
    :return:
    """

    # 通过聚类训练出词汇分类器
    #wc = WordCluster()
    #wc.fit(input_dir)
    #save(wc, output_dir + "wc.dump")
    wc = load(output_dir + "wc.dump")
    assert isinstance(wc, WordCluster)
    wc.fit(input_dir)
    #wc.display_words()
    save(wc, output_dir + "wc.dump")

    # 通过标注的样品，训练分类器
    dc = DocClassifier(wc)
    dc.fit()
    save(dc, output_dir + "dc.dump")


def predict(image_file, classifier_dump_file="/tmp/dc.dump"):
    """
    预测图像分类
    :param image_file: 待预测的文件
    :param classifier_dump_file: 训练好的分类器
    :return:
    """
    dc = load(classifier_dump_file)
    assert isinstance(dc, DocClassifier)
    dc.predict(image_file)

if __name__ == '__main__':

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hfps:", ["help","fit","predict","sample="])
    except getopt.error, msg:
        print msg
        print "for help use --help"
        sys.exit(2)
        # process options
    for o, a in opts:
        if o in ("-h", "--help"):
            print __doc__
            sys.exit(0)
            # process arguments

    #img = Image.open('/Users/caoyangmin/Documents/code/21.jpg').convert('L')
    #img = numpy.array(img, 'uint8')
    #features = get_features_from_image(img)
    #i = 0

    fit(u'/Users/caoyangmin/Documents/code/odr-samples')
