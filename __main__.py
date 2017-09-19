# coding=utf-8
from odr import WordCluster, DocClassifier
import cPickle as pickle

def save(var, name):
    file = open(name, 'wb')
    pickle.dump(var, file, protocol=1)
    file.close()


def load(name):
    file = open(name, 'rb')
    var = pickle.load(file)
    file.close()
    return var


def fit(input_dir, output_dir="/tmp/"):
    """
    训练
    :param input_dir: 样板所在所在目录
    :param output_dir: 训练结果存放路径
    :return:
    """

    # 通过聚类训练出词汇分类器
    wc = WordCluster()
    wc.fit(input_dir)
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

    print u"Usage: main fit samples_dir\nmain fit samples_dir"

    fit()
