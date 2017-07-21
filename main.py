#coding=utf8
import codecs
from bs4 import BeautifulSoup
import jieba
import numpy as np
from utils import *
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from gensim import corpora, models, similarities
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import spectral_clustering
from sklearn.cluster import AgglomerativeClustering
from time import time
import jieba.posseg as pseg

logfile = codecs.open("result.log", "w", "utf-8")

class clf_result:
    name = ""
    purity = np.array([])
    inversepurity = np.array([])
    f_measure = np.array([])
    mean_purity = .0
    mean_inversepurity = .0
    mean_f_measure = .0
    def calc_mean(self):
        self.mean_purity = self.purity.mean()
        self.mean_inversepurity = self.inversepurity.mean()
        self.mean_f_measure = self.f_measure.mean()
        

def cmppurity(self,other):  
        if self.mean_purity < other.mean_purity:  
            return -1  
        elif self.mean_purity==other.mean_purity:  
            return 0  
        else:  
            return 1

def cmpinversepurity(self,other):  
        if self.mean_inversepurity < other.mean_inversepurity:  
            return -1  
        elif self.mean_inversepurity==other.mean_inversepurity:  
            return 0  
        else:  
            return 1

def cmpfmeasure(self,other):  
        if self.mean_f_measure < other.mean_f_measure:  
            return -1  
        elif self.mean_f_measure==other.mean_f_measure:  
            return 0  
        else:  
            return 1

def context(segmented,word_windows_size,tags,tag_windows_size):
    context = []
    i = 1
    for word,tag in zip(segmented,tags):
      #  print i
        i += 1
        index = word.index(keyword)
      #  print "index",index
        begin = index - word_windows_size
        end = index + word_windows_size
        if begin < 0:
            begin = 0
        if end > len(word)-1:
            end = len(word)-1
        
        
        tag_begin = index - tag_windows_size
        tag_end = index + tag_windows_size
        if tag_begin < 0:
            tag_begin = 0  
        if tag_end > len(word)-1:
            tag_end = len(word)-1      
        x={}
        for i in range(begin,end):
            if i == index:
                continue
            x[word[i]] = 1.0
    #        print word[i],
        for i in range(tag_begin,tag_end):
            x[tag[i]] = 1.0
        context.append(x)
     #   print '\n'
    vectorizer = DictVectorizer()
    context = vectorizer.fit_transform(context)
    return context

def lda(data,n_topics):
 #   b=np.loadtxt('model-final.theta')        
    dictionary = corpora.Dictionary(segmented)
    corpus = [dictionary.doc2bow(text) for text in segmented]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lda = models.ldamodel.LdaModel(corpus_tfidf, id2word=dictionary, iterations=500, num_topics=n_topics)
    topics =[lda[c] for c in corpus_tfidf]
    dense = np.zeros((len(topics),n_topics),float)
    for ti,t in enumerate(topics):
        for tj,v in t:
            dense[ti,tj] = v
    return dense

def bench_cluster(mapp,estimator, name, data,labels,n_classes):
    t0 = time()
    estimator.fit(data)
    if name.count('ap')==0 and name.count('ms')==0:
        n_clusters = estimator.n_clusters
    else:
        n_clusters = estimator.cluster_centers_.shape[0]
    #print labels
    #print estimator.labels_
    
    purity = calc_purity(estimator.labels_,labels,n_clusters,n_classes)
    inversepurity = calc_inversepurity(estimator.labels_,labels,n_clusters,n_classes)
    f_measure = calc_fmeasure(estimator.labels_,labels,n_clusters,n_classes)
    
    if mapp.has_key(name):
        clf_res = mapp[name]
    else:
        clf_res = clf_result()
        clf_res.name = name
        mapp[name] = clf_res
    
    clf_res.purity = np.append(clf_res.purity,purity)
    clf_res.inversepurity = np.append(clf_res.inversepurity,inversepurity)
    clf_res.f_measure = np.append(clf_res.f_measure,f_measure)
    
    logfile.write('%s'% (name).center(15)+('%.2fs'%(time() - t0)).center(10)+
                  ('%.3f'%purity).center(10)+
                  ('%.3f'%inversepurity).center(10)+
                  ('%.3f'%f_measure).center(10)+'\n')
def kmeans_cluster(mapp,name,data,n_classes,labels):
    clf = KMeans(init='k-means++', n_clusters=n_classes, n_init=10)
    bench_cluster(mapp,clf, name, data,labels,n_classes)
    
def affinity_propagation(mapp,name,data,n_classes,labels):
    clf = AffinityPropagation()
    bench_cluster(mapp,clf, name, data,labels,n_classes)

def mean_shift(mapp,name,data,n_classes,labels):
    clf = MeanShift()
    bench_cluster(mapp,clf, name, data,labels,n_classes)
    
def spectral_clustering(mapp,name,data,n_classes,labels):
    clf = spectral_clustering(n_clusters=n_classes, eigen_solver='arpack')
    bench_cluster(mapp,clf, name, data,labels,n_classes)    

def hierarchical_clustering(mapp,name,data,n_classes,labels):
    clf = AgglomerativeClustering(n_clusters=n_classes,linkage='ward')
    bench_cluster(mapp,clf, name, data,labels,n_classes)  


def prin_final_result(mapp):
    files = []
    purity_logfile = codecs.open("purity_result.log", "w", "utf-8")
    inversepurity_logfile = codecs.open("inversepurity_result.log", "w", "utf-8")
    fmeasurepurity_logfile = codecs.open("fmeasure_result.log", "w", "utf-8")
    files.append(purity_logfile)
    files.append(inversepurity_logfile)
    files.append(fmeasurepurity_logfile)
    for filea in files:
        filea.write('final:\n')
        filea.write(79 * '_'+'\n')
        filea.write('method'.center(15)+'purity'.center(10)+'inverse'.center(10)
                  +'f_measure'.center(10)+'\n')
        
    clflist = []
    for i in mapp:
        clflist.append(mapp[i])
        mapp[i].calc_mean()
    clflist.sort(cmp=cmppurity,reverse=True)
    for i in clflist:
        purity_logfile.write('%s'% (i.name).center(15)+
                ('%.3f'%i.mean_purity).center(10)+
                ('%.3f'%i.mean_inversepurity).center(10)+
                ('%.3f'%i.mean_f_measure).center(10)+'\n')        
        
    clflist.sort(cmp=cmpinversepurity,reverse=True)
    for i in clflist:
        inversepurity_logfile.write('%s'% (i.name).center(15)+
                ('%.3f'%i.mean_purity).center(10)+
                ('%.3f'%i.mean_inversepurity).center(10)+
                ('%.3f'%i.mean_f_measure).center(10)+'\n')

    clflist.sort(cmp=cmpfmeasure,reverse=True)
    for i in clflist:
        fmeasurepurity_logfile.write('%s'% (i.name).center(15)+
                ('%.3f'%i.mean_purity).center(10)+
                ('%.3f'%i.mean_inversepurity).center(10)+
                ('%.3f'%i.mean_f_measure).center(10)+'\n')


if __name__ == '__main__':
    result_mapp = {}
            
    logfile.write(79 * '_'+'\n')
    logfile.write('method'.center(15)+'time'.center(10)+'purity'.center(10)+'inverse'.center(10)
                  +'f_measure'.center(10)+'\n')
    sample_files = os.listdir('./raw/') 
  #  sample_files = ['sample.xml','sample2.xml'] 
    for sample_file in sample_files:
        logfile.write(sample_file[:-4].decode('gbk')+'\n')
        fh =codecs.open('./raw/'+sample_file, "r", "utf-8")
        text = fh.read()
        text = text.replace("<head>",u"|")
        text = text.replace("</head>",u"|")
        soup = BeautifulSoup(text)
        
        
        keyword = soup.find("lexelt")["item"]
        n_classes = int(soup.find("lexelt")["snum"])
        
        #contain all instances
        instances = soup.findAll('instance')
        jieba.suggest_freq(keyword, True)
        jieba.add_word(keyword)
        segmented = []
        tags = []
        for instance in instances:
            tag = []
            temp = []
            words = pseg.cut(instance.getText())
            for w in words:
                if w.word == '\r\n' or w.word == u'|':
                    continue
                temp.append(w.word)
                tag.append(w.flag)
            segmented.append(temp)
            tags.append(tag)
        writetofile('./parsed/'+sample_file[:-3]+'txt',segmented)
        writetofile('./parsed/tag_'+sample_file[:-3]+'txt',tags)

        labels=np.zeros(len(segmented),dtype=np.int32)
        #print 'len labels: %d' % len(labels)
        fh =codecs.open('./answer/'+sample_file, "r", "utf-8")
        text = fh.read()
        
        soup = BeautifulSoup(text)  
        
        #contain all instances
        senses = soup.findAll('sense')
        
        for i in range(len(senses)):
            instances = senses[i].findAll('instance')
            for instance in instances:
                number = int(instance["id"])
            #    print number-1
                labels[number-1] = i
                
        for n_topics in range(5,30):
            data = lda(segmented,n_topics)
            reduced_data = PCA(n_components=0.7).fit_transform(data)
            kmeans_cluster(result_mapp,'lda:%d k+'%n_topics,data,n_classes,labels)
            affinity_propagation(result_mapp,'lda:%d ap'%n_topics,data,n_classes,labels)
            mean_shift(result_mapp,'lda:%d ms'%n_topics,data,n_classes,labels)
            hierarchical_clustering(result_mapp,'lda:%d hc'%n_topics,data,n_classes,labels)
            kmeans_cluster(result_mapp,'lda:%d k+pca'%n_topics,reduced_data,n_classes,labels)
            affinity_propagation(result_mapp,'lda:%d appca'%n_topics,reduced_data,n_classes,labels)
            mean_shift(result_mapp,'lda:%d mspca'%n_topics,reduced_data,n_classes,labels)
            hierarchical_clustering(result_mapp,'lda:%d hcpca'%n_topics,reduced_data,n_classes,labels)
        for word_windows_size in range(2,12):
            for tag_windows_size in range(-1,4):
                data = context(segmented,word_windows_size,tags,tag_windows_size)
                reduced_data = PCA(n_components=0.7).fit_transform(data.toarray())
                kmeans_cluster(result_mapp,'c:w%dt:%d k+' % (word_windows_size,tag_windows_size),data,n_classes,labels)
                affinity_propagation(result_mapp,'c:w%dt:%d ap' % (word_windows_size,tag_windows_size),data,n_classes,labels)
                mean_shift(result_mapp,'c:w%dt:%d ms' % (word_windows_size,tag_windows_size),data.toarray(),n_classes,labels)
                hierarchical_clustering(result_mapp,'c:w%dt:%d hc' % (word_windows_size,tag_windows_size),data.toarray(),n_classes,labels)
                kmeans_cluster(result_mapp,'c:w%dt:%d k+pca' % (word_windows_size,tag_windows_size),reduced_data,n_classes,labels)
                affinity_propagation(result_mapp,'c:w%dt:%d appca' % (word_windows_size,tag_windows_size),reduced_data,n_classes,labels)
                mean_shift(result_mapp,'c:w%dt:%d mspca' % (word_windows_size,tag_windows_size),reduced_data,n_classes,labels)
                hierarchical_clustering(result_mapp,'c:w%dt:%d hcpca' % (word_windows_size,tag_windows_size),reduced_data,n_classes,labels)
        logfile.write(79 * '_'+'\n')
        '''
        for n_topics in range(5,30):
            lda_data = lda(segmented,n_topics)
            for word_windows_size in range(2,12):
                for tag_windows_size in range(-1,4):
                    data = context(segmented,word_windows_size,tags,tag_windows_size)
                    data = np.hstack((data.toarray(),lda_data))
                    reduced_data = PCA(n_components=0.7).fit_transform(data)
                    kmeans_cluster(result_mapp,'lda:%dc:w%dt:%d k+' % (n_topics,word_windows_size,tag_windows_size),data,n_classes,labels)
                    affinity_propagation(result_mapp,'lda:%dc:w%dt:%d ap' % (n_topics,word_windows_size,tag_windows_size),data,n_classes,labels)
                    mean_shift(result_mapp,'lda:%dc:w%dt:%d ms' % (n_topics,word_windows_size,tag_windows_size),data,n_classes,labels)
                    hierarchical_clustering(result_mapp,'lda:%dc:w%dt:%d hc' % (n_topics,word_windows_size,tag_windows_size),data,n_classes,labels)
                    kmeans_cluster(result_mapp,'lda:%dc:w%dt:%d k+pca' % (n_topics,word_windows_size,tag_windows_size),reduced_data,n_classes,labels)
                    affinity_propagation(result_mapp,'lda:%dc:w%dt:%d appca' % (n_topics,word_windows_size,tag_windows_size),reduced_data,n_classes,labels)
                    mean_shift(result_mapp,'lda:%dc:w%dt:%d mspca' % (n_topics,word_windows_size,tag_windows_size),reduced_data,n_classes,labels)
                    hierarchical_clustering(result_mapp,'lda:%dc:w%dt:%d hcpca' % (n_topics,word_windows_size,tag_windows_size),reduced_data,n_classes,labels)
        '''                    
    prin_final_result(result_mapp)