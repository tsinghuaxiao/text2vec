import gzip
import json
import shelve
import time
import pickle, pprint
import csv
import numpy as np
import nltk
import re
import os
import gensim
from collections import Counter
from gensim.models import doc2vec
from bs4 import BeautifulSoup
from collections import Counter
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
TaggededDocument = gensim.models.doc2vec.TaggedDocument


start = time.time()


def readData(path):
    g = gzip.open(path, 'r')
#     print(g)
    for l in g:
#         yield json.loads(l)
        yield eval(l)


# 读取BoughtTogetherDic.pickle
def readBoughtTogetherDic(createPathBoughtTogetherDic):
    with open(createPathBoughtTogetherDic, 'rb') as f:
        boughtTogetherDic = pickle.load(f)
        return boughtTogetherDic

# 获取对应100个产品及其boughtTogether的title或者ReviewText
def createTitleReview(style, sourcePath, save_path):
    k = 0
    for data in readData(sourcePath):
        if style in data.keys():
            tempdic = {}
            with open(save_path, 'ab') as f:
                tempdic = {data['asin']:data[style]}
                pickle.dump(tempdic, f)
                k += 1
        print(k)

def readTitleReview(save_path):
    data = []
    with open(save_path, 'rb') as f:
        while True:
            try:
                line = pickle.load(f)
                data.append(line)
            except EOFError:
                break
    return data

# 合并相同产品的reviewText
def mergeReview(save_path_review):
    reviewdic = {}
    data_Review = readTitleReview(save_path_review)
    print("加载完毕!")
    i = 0
    
    for dic in data_Review:
        print(i)
        for key, value in dic.items():
            reviewdic[key] = reviewdic.get(key, '') + ' ' + value
        i += 1
    with open('H:/dataset/createFull/textvectemp/reviewdicMerge.pickle', 'wb') as f:
        pickle.dump(reviewdic, f)
    
#         if list(dic.keys())[0] not in reviewdic.keys():
#             reviewdic[list(dic.keys())[0]] = list(dic.values())[0]
#         else:
#             reviewdic[list(dic.keys())[0]] = list(reviewdic.values())[0] + ' ' + list(dic.values())[0]
#     f.close()
#     with open('H:/dataset/textvectemp/reviewdicMerge.txt', 'w') as f:
#         f.write(str(reviewdic))

# 合并相同产品的title和reviewText    
def mergeTitleReview(save_path_meta, save_path_reviewMerge, save_path_text):
    textdic = {}
    data_Title = readTitleReview(save_path_meta)
    data_ReviewMerge = readTitleReview(save_path_reviewMerge)
    print("加载完毕!")
    i = 0

    for data_title in data_Title:
        print(i)
        i += 1
        key_title = list(data_title.keys())[0]
        value_title = list(data_title.values())[0]
        if key_title in data_ReviewMerge[0].keys():
            textdic[key_title] = value_title + '. ' + data_ReviewMerge[0][key_title]
    f = open(save_path_text, 'wb')
    pickle.dump(textdic, f)
    f.close()
    
    
#     f = open('H:/dataset/textvectemp/textdic.txt', 'w')
#     f.write(str(textdic))


# 读取产品的asin
def read_asin(save_path_text):                
    asin = []
    with open(save_path_text, 'rb') as f:
        line = pickle.load(f)
    for key, _ in line.items():
        asin.append(key)
    return asin


def read_trainText(path):
    with open(path, 'rb') as f:
        line = pickle.load(f)
    return line

# 分割句子   
def sentence_segment(text):
        raw_text = BeautifulSoup(text).get_text()
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        raw_sentences = tokenizer.tokenize(raw_text)
        sentences = []
        for sent in raw_sentences:
            sentences.append(re.sub('[^a-zA-Z]', ' ', sent.lower().strip()).split())
        return sentences

def train(x_train, vector_size=100, epoch_num=1):
 
#     x_train = [doc2vec.TaggedDocument(sentence, 'tag' + str(i)) for i, sentence in enumerate(x_train)]
    model_dm = Doc2Vec(x_train, min_count=1, window=5, vector_size=vector_size, sample=1e-3, negative=5, workers=4)
    print("Training......")
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    print("Training over!")
    model_dm.save('H:/dataset/createFull/model/title_review_gensim') ##模型保存的位置
    
    return model_dm

# 得到商品的描述句子向量
def getTextVec(model_dm, item_asin):
    inferred_vector_dm = model_dm.docvecs[item_asin]
    return inferred_vector_dm


# 制作商品描述文本向量pickle数据
def createTextPickle(model_dm, item_asin, createPathTextVec):
    with open(createPathTextVec, 'wb') as f:
        temp_dic = {}
        i = 0
        for asin in item_asin:
            vec_asin = asin
            vec_feature = getTextVec(model_dm, asin)
            temp_dic.update({vec_asin:vec_feature})
            if (i+1) % 200 == 0:
                print("第" + str(i) + "行")
            i += 1
        print("第" + str(i) + "行")
        pickle.dump(temp_dic, f)

def readTextVec(createPathTextVec):
    with open(createPathTextVec, 'rb') as f:
        TextVecDic = pickle.load(f)
        return TextVecDic


def readWordCount(asin, titleWordDic, reviewTextWordDic):
    reviewTextCount = 0
    
    title = titleWordDic[asin]
    sentences_title = sentence_segment(title)
    titleCount = len(sentences_title[0])
    
    
    review = reviewTextWordDic[asin]
    sentences_review = sentence_segment(review)
    for sentence in sentences_review:
        reviewTextCount += len(sentence)
    
    return titleCount, reviewTextCount

# 增加BoughtTogetherInfoDic.pickle
# {item:[textual word count with smallercase, textual word count 单词数量, text vector]}   
def addBoughtTogetherInfoDic(createPathTextVec, createPathBoughtTogetherDic, save_path_meta, save_path_reviewMerge, createPathBoughtTogetherInfo):
    boughtTogetherInfoDic = {}
    info = []
    
    titleWordDic = {}
    titleWord = readTitleReview(save_path_meta)
    for items in titleWord:
        titleWordDic[list(items.keys())[0]] = list(items.values())[0]
    print("Title加载完毕!")
    
    f = open(save_path_reviewMerge, 'rb')
    reviewTextWordDic = pickle.load(f)
    f.close()
    print("reviewMerge加载完毕!")
    
    textVecDic = readTextVec(createPathTextVec)
    boughtTogetherKeyDic = readBoughtTogetherDic(createPathBoughtTogetherDic)['bought_together']
    print("textVecDic加载完毕!")
    
    n = 0
    with open(createPathBoughtTogetherInfo, 'wb') as f:
        for _, value in boughtTogetherKeyDic.items():
            for boughtTogetherKey in value:
                if boughtTogetherKey in textVecDic.keys():
                    asin = boughtTogetherKey
                    titleCount, reviewTextCount = readWordCount(asin, titleWordDic, reviewTextWordDic)
                    textVec = textVecDic[asin]
                    info.append(titleCount)
                    info.append(reviewTextCount)
                    info.append(textVec)
                    boughtTogetherInfoDic[asin] = info
            print(n)
            n += 1
        pickle.dump(boughtTogetherInfoDic, f)

def readBoughtTogetherInfoDic(createPathBoughtTogetherInfo):
    with open(createPathBoughtTogetherInfo, 'rb') as f:
        infoDic = pickle.load(f)
    print(len(infoDic.keys()))


   
sourcePathmeta = 'H:/dataset/years2014/meta_Clothing_Shoes_and_Jewelry.json.gz'
sourcePathreview = 'H:/dataset/years2014/reviews_Clothing_Shoes_and_Jewelry.json.gz'
createPathTextVec = 'H:/dataset/createFull/textvec/TextVecDic.pickle'
createPathBoughtTogetherInfo = 'H:/dataset/createFull/BoughtTogetherDic/BoughtTogetherDicInfo.pickle'

save_path_meta = 'H:/dataset/createFull/textvectemp/metadic.pickle'
save_path_review = 'H:/dataset/createFull/textvectemp/reviewdic.pickle'
save_path_text = 'H:/dataset/createFull/textvectemp/textdic.pickle'
save_path_reviewMerge = 'H:/dataset/createFull/textvectemp/reviewdicMerge.pickle'
save_path_traintext = 'H:/dataset/createFull/textvectemp/traintext.pickle'

# 读取BoughtTogetherDic.pickle
createPathBoughtTogetherDic = 'H:/dataset/createFull/BoughtTogetherDic/BoughtTogetherDic.pickle'
# boughtTogetherDic = readBoughtTogetherDic(createPathBoughtTogetherDic)

# 读取Title或者ReviewText.pickle
# createTitleReview('title', sourcePathmeta, save_path_meta)
# createTitleReview('reviewText', sourcePathreview, save_path_review)
# readTitleReview(save_path_review)

# 合并相同商品的ReviewText文本
# mergeReview(save_path_review)

# 合并相同商品的Title和ReviewText文本
# mergeTitleReview(save_path_meta, save_path_reviewMerge, save_path_text)

# dic = readTitleReview(save_path_reviewMerge)
# dic2 = readTitleReview(save_path_review)
# with open('H:/dataset/createFull/textvectemp/a.txt', 'w') as f:
#     f.write(str(dic[0]['0000031887']))
# with open('H:/dataset/createFull/textvectemp/b.txt', 'w') as f:
#     i = 0
#     for temp in dic2:
#         for k, v in temp.items():
#             if k == '0000031887':
#                 f.write(v)
#         i += 1
#         print(i)


# 切割句子
# data = read_trainText(save_path_text)
# j = 0
# with open('H:/dataset/createFull/textvectemp/trainText.pickle', 'ab') as f:
#     for key, text in data.items():
#         print(j)
#         j += 1
#         # 切割句子
#         text_cut = sentence_segment(text)
#         item_text = []
#         x_train = {}
#         for i, text in enumerate(text_cut):
#             item_text += text
#         x_train[key] = item_text
# #         document = TaggededDocument(item_text, tags=[key])
# #         x_train.append(document)
#         pickle.dump(x_train, f)

# 训练
# train_data = []
# with open('H:/dataset/createFull/textvectemp/trainText.pickle', 'rb') as f:
#     while True:
#         try:
#             line = pickle.load(f)
#             train_data.append(line)
#         except EOFError:
#             break
# x_train = []
# k = 0
# for dic in train_data:
#     print(k)
#     k += 1
#     document = TaggededDocument(list(dic.values())[0], tags=[list(dic.keys())[0]])
#     x_train.append(document)
# model_dm = train(x_train)


# 创建TextVecDic.txt
# 加载文本向量模型
# model_dm = Doc2Vec.load('H:/dataset/createFull/model/title_review_gensim')
# item_asin = read_asin(save_path_text)
# createTextPickle(model_dm, item_asin, createPathTextVec)

# TextVecDic = readTextVec(createPathTextVec)
# print(len(TextVecDic))
# print(TextVecDic['0000031887'].reshape((1,100)))

# addBoughtTogetherInfoDic(createPathTextVec, createPathBoughtTogetherDic, save_path_meta, save_path_reviewMerge, createPathBoughtTogetherInfo)
readBoughtTogetherInfoDic(createPathBoughtTogetherInfo)

end = time.time()
print('time:', end - start)
