# coding=utf-8
import os
import lmdb
import numpy
import cPickle as pickle
import sqlite3
__author__ = 'caoym'

class SubDB:
    def __init__(self, path, name):
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)
        self.env = lmdb.open(self.path,map_size=5000000000,max_dbs=1)
        self.txn = self.env.begin(write=True)
        self.pos = 0
        self.name = name

    def count(self):
        return self.env.stat()['entries']
    def getFeatures(self, start=0, size=-1):
        count = 0
        i = start
        while True:
            value = self.txn.get(str(i))
            if value is None:
                break

            i += 1
            data = pickle.loads(value)
            yield data
            count += 1
            if size >= 0 and count >= size:
                break
    def getFeature(self, key):
        count = 0

        value = self.txn.get(str(key))
        if value is None:
            return None

        data = pickle.loads(value)
        return data

    def clear(self):
        self.pos = 0
        db = self.env.open_db()
        self.txn.drop(db)
        self.env.sync(True)

    def addFeature(self, var):
        data = pickle.dumps(var)
        self.txn.put(str(self.pos), data)
        self.pos += 1
        if self.pos%100 ==0:
            print('addFeature',self.name,self.pos)

    def setFeature(self,key, var):
        data = pickle.dumps(var)
        self.txn.put(str(key), data)
        if self.pos%100 ==0:
            print('setFeature',self.name,key)

    def flush(self):
        self.txn.commit()
        self.env.sync(True)
        #self.env.close()

class DB:
    def __init__(self, path, data_type):
        self.path = path
        self.data_type = data_type

    def getSubDBs(self):
        dir = os.listdir(self.path + "/" + self.data_type)
        for type in dir:
            yield SubDB(self.path + "/" + self.data_type + "/" + type, type)

    def getFeatures(self):
        for i in self.getSubDBs():
            yield i.getFeatures()

    def newSubDB(self, name, clear):
        subdb = SubDB(self.path + "/" + self.data_type + "/" + name, name)
        if (clear):
            subdb.clear()
        return subdb

    def getSubDB(self, name):
        subdb = SubDB(self.path + "/" + self.data_type + "/" + name, name)
        return subdb

    def getItemById(self, id):
        for i in self.getSubDBs():
            if i.count() > id:
                return i.getFeatures(id,1).next();
            else:
                id -= i.count()
        return None;

    def getSubDBById(self, id):
        for i in self.getSubDBs():
            if i.count() > id:
                return i;
            else:
                id -= i.count()
        return None;

    def count(self):
        count = 0
        for i in self.getSubDBs():
            count += i.count()
        return  count

