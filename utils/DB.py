# coding=utf-8
__author__ = 'caoym'
from peewee import *
from playhouse.sqlite_ext import SqliteExtDatabase
import cPickle as pickle
from StringIO import StringIO
from PIL import Image

db = SqliteDatabase("D:/data/trains.db" )
db.get_conn().text_factory = str

class PickleField(BlobField):

    def db_value(self, value):
        return buffer(pickle.dumps(value,protocol=1))

    def python_value(self, value):
        """Convert the database value to a pythonic value."""
        return value if value is None else pickle.loads(str(value))

class ImageField(BlobField):

    def db_value(self, value):
        buf = StringIO()
        value = value.convert('RGB')
        value.save(buf, "PNG")
        return buffer(buf.getvalue())

    def python_value(self, value):
        return  Image.open(StringIO(value));

class Feature(Model):
    '''
    特征表
    '''
    ori = PickleField() #
    img = ImageField() #
    label = CharField(index=True)
    docname = CharField(index=True)
    entropy = FloatField(index=True,default=0)
    ignore = BooleanField(index=True,default=0)
    class Meta:
        database = db

class PcaModel(Model):
    feature = ForeignKeyField(Feature)
    pca = PickleField()
    class Meta:
        database = db

class DescriptorModel(Model):
    feature = ForeignKeyField(Feature)
    lv1 = PickleField()
    lv2 = PickleField()
    class Meta:
        database = db

class TrainingResult(Model):
    data = PickleField(null=True)
    name = CharField(index=True,unique=True)
    class Meta:
        database = db

class Words(Model):
    chi = FloatField(index=True,default=0)
    idf = FloatField(index=True,default=0)
    ignore = BooleanField(index=True,default=0)
    class Meta:
        database = db

class Vocabulary(Model):
    lv1 = IntegerField(index=True)
    lv2 = IntegerField(index=True)
    feature = ForeignKeyField(Feature)
    class Meta:
        database = db

class SubWords(Model):
    ignore = BooleanField(index=True,default=0)
    label = CharField(index=True)
    class Meta:
        database = db

class SubVocabulary(Model):
    word = ForeignKeyField(SubWords)
    feature = ForeignKeyField(Feature)
    ignore = BooleanField(index=True,default=0)
    class Meta:
        database = db

class CoreSample(Model):
    feature = ForeignKeyField(Feature)
    class Meta:
        database = db

