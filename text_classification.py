import io, os
import re as re
import zipfile as zipfile
import sys
import random
import math 
mytextzip = ''
docList=[]
idx_ID=1
author=0
with zipfile.ZipFile('30Columnists.zip') as z:
  for zipinfo in z.infolist():
    mytextzip = ''
    if zipinfo.filename.endswith('.txt') and re.search('raw_texts', zipinfo.filename):
       with z.open(zipinfo) as f:
          textfile = io.TextIOWrapper(f, encoding='cp1254', newline='')
          for line in textfile:
            if len(line.strip()): mytextzip += ' ' + line.strip()
          document = {
            'id': str(idx_ID),
            'text': mytextzip,
            'author':author
          }
          docList.append(document)
          if idx_ID % 50 == 0:
            author+=1
          idx_ID+=1
          


# TOKENIZATION

# Non-breaking to normal space
NON_BREAKING = re.compile(u"\s+"), " "
# Multiple dot
MULTIPLE_DOT = re.compile(u"\.+"), " "
# Merge multiple spaces.
ONE_SPACE = re.compile(r' {2,}'), ' '
# 2.5 -> 2.5 - asd. -> asd . 
DOT_WITHOUT_FLOAT = re.compile("((?<![0-9])[\.])"), r' '
# 2,5 -> 2,5 - asd, -> asd , 
COMMA_WITHOUT_FLOAT = re.compile("((?<![0-9])[,])"), r' '
# doesn't -> doesn't  -  'Something' -> ' Something '
QUOTE_FOR_NOT_S = re.compile("[\']"), r' '
AFTER_QUOTE_SINGLE_S = re.compile("\s+[s]\s+"), r' '
# Extra punctuations "!()
NORMALIZE = re.compile("([\–])"), r'-'
EXTRAS_PUNK = re.compile("([^\'\.\,\w\s\-\–])"), r' '

STOP_WORDS_LIST=['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']
STOP_WORDS=re.compile(r'\b(?:%s)\b[^\-]' % '|'.join(STOP_WORDS_LIST)),r" "

REGEXES = [
    STOP_WORDS,
    NON_BREAKING,
    MULTIPLE_DOT,
    DOT_WITHOUT_FLOAT,
    COMMA_WITHOUT_FLOAT,
    QUOTE_FOR_NOT_S,
    AFTER_QUOTE_SINGLE_S,
    NORMALIZE,
    EXTRAS_PUNK,
    ONE_SPACE
]

def pre_porcess_tokenize_sentence(sentence):
  sentence = sentence.lower()
  for regexp, subsitution in REGEXES:
    sentence = regexp.sub(subsitution, sentence)   
  return sentence

import time
start_time = time.time()

tokenizedList=[]
for doc in docList:
  tokenizedText = pre_porcess_tokenize_sentence(doc['text'])
  tokens = tokenizedText.split(' ')
  del tokens[0]
  del tokens[len(tokens)-1]
  tokenizedList.append(tokens)

elapsed_time = time.time() - start_time

print("Tokenize: "+str(elapsed_time))
#print(tokenizedList)

# DOCUMENT VECTOR
start_time = time.time()
wordsFreqMatrix={}

docIdx=0
for wordLists in tokenizedList:
  for word in wordLists:
    if word in wordsFreqMatrix.keys():
      wordsFreqMatrix[word][docIdx] += 1
    else:
      wordsFreqMatrix[word]=[0 for i in range(0,1500)]
      wordsFreqMatrix[word][docIdx] += 1
  docIdx += 1

doc2vec={i:[row[i] for row in wordsFreqMatrix.values()] for i in range(docIdx)}

elapsed_time = time.time() - start_time
print("Document Vector: ",str(elapsed_time))

# Cosine similarity

def FindColMinMax(items): 
  n = len(items[0]) 
  minima = [sys.maxsize for i in range(n)] 
  maxima = [-sys.maxsize -1 for i in range(n)] 
    
  for item in items.values(): 
    for f in range(len(item)): 
      if (item[f] < minima[f]): 
        minima[f] = item[f] 

      if (item[f] > maxima[f]): 
        maxima[f] = item[f] 
  
  return minima,maxima

def InitializeMeans(items, k, cMin, cMax): 
  # Initialize means to random numbers between 
  # the min and max of each column/feature     
  f = len(items[0]) # number of features 
  means = [[0 for i in range(f)] for j in range(k)]
    
  for mean in means: 
    for i in range(len(mean)): 
      # Set value to a random float 
      # (adding +-1 to avoid a wide placement of a mean) 
      mean[i] = random.uniform(cMin[i]+1, cMax[i]-1) 
  
  return means 

def CossineSimilarity(x, y): 
  multiplication=[a*b for a,b in zip(x,y)]
  totalTop=sum(multiplication)
  squareA=sum([a*a for a in x])
  squareB=sum([a*a for a in y])
  rootSumm=math.sqrt(squareA)+math.sqrt(squareB)
  distance=totalTop/rootSumm

  return math.sqrt(distance) 

def UpdateMean(n,mean,item): 
  for i in range(len(mean)): 
    m = mean[i] 
    m = (m*(n-1)+item[i])/float(n) 
    mean[i] = round(m, 3) 
    
  return mean

def Classify(means,item): 
    # Classify item to the mean with minimum distance     
    minimum = sys.maxsize 
    index = -1 
  
    for i in range(len(means)): 
  
        # Find distance from item to mean 
        dis = CossineSimilarity(item, means[i]) 
        if (dis < minimum): 
            minimum = dis 
            index = i 
    return index 

def CalculateMeans(k,items,maxIterations=10): 
  
    # Find the minima and maxima for columns 
    cMin, cMax = FindColMinMax(items) 
      
    # Initialize means at random points 
    means = InitializeMeans(items,k,cMin,cMax) 
      
    # Initialize clusters, the array to hold 
    # the number of items in a class 
    clusterSizes= [0 for i in range(len(means))] 
  
    # An array to hold the cluster an item is in 
    belongsTo = [0 for i in range(len(items))] 
    first=True
    # Calculate means 
    for e in range(maxIterations): 

        # If no change of cluster occurs, halt 
        noChange = True 
        
        for i in range(len(items)): 
  
            item = items[i] 
  
            # Classify item into a cluster and update the 
            # corresponding means.         
            index = Classify(means,item) 
  
            clusterSizes[index] += 1 
            cSize = clusterSizes[index] 
            means[index] = UpdateMean(cSize,means[index],item) 
  
            # Item changed cluster 
            if(index != belongsTo[i]): 
                noChange = False 
            if not first:
              clusterSizes[belongsTo[i]]-=1
            belongsTo[i] = index 
        first=False
        # Nothing changed, return 
        if (noChange): 
            break 
  
    return means

def FindClusters(means,items): 
    clusters = [[] for i in range(len(means))] # Init clusters 
    for key,item in items.items(): 
  
        # Classify item into a cluster 
        index = Classify(means,item) 
  
        # Add item to cluster 
        clusters[index].append(key) 
  
    return clusters 
  

means=CalculateMeans(4,doc2vec)

clusters= FindClusters(means,doc2vec)

for i in range(len(clusters)):
  print('%d. Class Documents: \n' % (i))
  for j in clusters[i]:
    print(docList[j])