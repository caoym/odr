# coding=utf-8
__author__ = 'caoym'

if __name__ == '__main__':
    from wordseg.wordseg import WordSegment
    f = open("D:\\cloud\\test.txt")
    doc = unicode(f.read(),'utf8')
    ws = WordSegment(doc, max_word_len=5, min_freq=0.00005, min_entropy=1.0, min_aggregation=300)
    print ' '.join(map(lambda w: '%s:%f'%w, ws.word_with_freq))
    print ' '.join(ws.words)
    print 'average len: ', ws.avg_len
    print 'average frequency: ', ws.avg_freq
    print 'average left entropy: ', ws.avg_left_entropy
    print 'average right entropy: ', ws.avg_right_entropy
    print 'average aggregation: ', ws.avg_aggregation