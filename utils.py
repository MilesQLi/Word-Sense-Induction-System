#coding=utf8
import codecs
import os 
import numpy as np



def writetofile(name, contents):
    length = len(contents)
    file = codecs.open(name, "w", "utf-8")
    file.write("%d\n" % length)
    
    for content in contents:
        #content.remove(u'\r\n')
        for word in content:
            file.write("%s " % word)
        file.write("\n")
        
def calc_purity(clusters,classes,n_clusters,n_classes):
    total = 0.
    maxi = 0.
    temp = 0.
    for i in range(n_clusters):
        maxi = 0.
        for j in range(n_classes):
            temp = 0.
            for k in range(len(clusters)):
                if clusters[k] == i and classes[k] == j:
                    temp += 1.
            if temp > maxi:
                maxi = temp
        total += maxi
    total /= len(clusters)
    return total
def calc_inversepurity(clusters,classes,n_clusters,n_classes):
    total = 0.
    maxi = 0.
    temp = 0.
    for i in range(n_classes):
        maxi = 0.
        for j in range(n_clusters):
            temp = 0.
            for k in range(len(clusters)):
                if classes[k] == i and clusters[k] == j:
                    temp += 1.
            if temp > maxi:
                maxi = temp
        total += maxi
    total /= len(clusters)
    return total


def calc_precision(classes,clusters,i,j):
    totalj = 0.
    totalinter = 0.
    for k in range(len(clusters)):
        if clusters[k] == j:
            totalj += 1.
            if classes[k] == i:
                totalinter += 1.
   # print "precision:%f" % (totalinter / totalj)
    return totalinter / totalj
        
def calc_recall(classes,clusters,i,j):
    totali = 0.
    totalinter = 0.
    for k in range(len(classes)):
        if classes[k] == i:
            totali += 1.
            if clusters[k] == j:
                totalinter += 1.
   # print "recall:%f" % (totalinter / totali)
    return totalinter / totali

def calc_fmeasureij(classes,clusters,i,j):
    pre = calc_precision(classes,clusters,i,j)
    rec = calc_recall(classes,clusters,i,j)
    if pre == 0 and rec == 0:
        return 0
    return (2 * pre * rec) / (pre + rec)


def calc_fmeasure(classes,clusters,n_classes,n_clusters):
    final = 0.
    for i in range(n_classes):
        number = classes.tolist().count(i)
        temp = []
        for j in range(n_clusters):
            temp.append(calc_fmeasureij(classes,clusters,i,j))
        temp = np.array(temp)
       # print "i:%d max:%f number:%d" % (i, temp.max(), number)
        final += number * temp.max()
    return final / len(classes)












