

import stanfordnlp
#from stanfordcorenlp import StanfordCoreNLP
#from pycorenlp import StanfordCoreNLP

#stanfordnlp.download('en')
#path = r'/Users/yuanlingou/Desktop/stanford-corenlp-4.0.0'
#nlp = StanfordCoreNLP(path,lang='en')


# 单个文本内容
def single_txt(path_file):
    with open(path_file, 'r')as f:
        lst = f.readlines()[1:]
        content = ''.join(lst)
        content = content.replace('\n', ' ')
    return content

class Doc:
    def __init__(self):
        self.words = []
        self.pos_tags =[]
        self.dependency_relation_triplets = []

# 初始文本特征生成器
nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', lang='en')
def gen_stanford_org_feature(doc):
    #stanfordnlp.download("en")
    doc = nlp(doc)
    sentences = doc.sentences
    DOC = Doc()
    for sent in sentences:
        words, pos_tags, dependency_relation_triplets = [], [], []
        for word in sent.words:
            words.append(word.text)
            pos_tags.append(word.pos)
            relation = word.dependency_relation
            governor = sent.words[word.governor - 1].text if word.governor > 0 else 'root'
            dependency_relation_triplets.append((relation, governor, word.text))
        DOC.words.append(words)
        DOC.pos_tags.append(pos_tags)
        DOC.dependency_relation_triplets.append(dependency_relation_triplets)
    return DOC

# 生成char串
def gen_char_str(words):
    chars = []
    for sent_words in words:
        chars.append(''.join(sent_words))
    return chars

import string
symbols = string.punctuation
def filter_special_symbol(lst):
    for item in lst:
        if item in symbols:
            return False
    return True

# ngram生成
def ngram_split(lst, n):
    ngrams = []
    for i in range(len(lst)-n):
        ngram = lst[i:i+n]
        if filter_special_symbol(ngram):
            ngrams.append(str(ngram))
    return ngrams

word_ngram_count_dict = {}
pos_tag_ngram_count_dict = {}
char_ngram_count_dict = {}
dependency_relation_triplet_count_dict = {}
# 统计ngram次数
def count_ngram_tf(lst, dic):
    for i in lst:
        if i in dic.keys():
            dic[i] +=1
        else:
            dic[i] = 1

# 生成ngram词典
def gen_ngram_dict(dic):
    ngram_dict=set()
    for k, v in dic.items():
        if int(v)>=5:
            ngram_dict.add(k)
    return ngram_dict

# 单个文本特征处理
def single(path, file, dir_path):
    def _del_file(file):
        file_path = os.path.join(dir_path, file)
        if os.path.exists(file_path):
            os.remove(file_path)
    file_pre = file.split('.')[0]
    org_word_ngram_file = '{0}-word-ngram.txt'.format(file_pre)
    org_pos_ngram_file = '{0}-pos-ngram.txt'.format(file_pre)
    org_char_ngram_file = '{0}-char-ngram.txt'.format(file_pre)
    org_dependency_relation_triplets_file = '{0}-dependency-relation-triplets.txt'.format(file_pre)
    _del_file(org_word_ngram_file)
    _del_file(org_pos_ngram_file)
    _del_file(org_char_ngram_file)
    _del_file(org_dependency_relation_triplets_file)
    words_ngram_file = open(os.path.join(dir_path, org_word_ngram_file), 'w')
    pos_tags_ngram_file = open(os.path.join(dir_path, org_pos_ngram_file), 'w')
    chars_ngram_file = open(os.path.join(dir_path, org_char_ngram_file), 'w')
    dependency_relation_triplets_file = open(os.path.join(dir_path, org_dependency_relation_triplets_file), 'w')

    # 1、获取文件数据
    doc = single_txt(path)
    # 2、文本特征生成
    DOC = gen_stanford_org_feature(doc)
    # 3、生成文本char串
    chars = gen_char_str(DOC.words)
    DOC.chars = chars
    # 4、生成ngram
    for sent_words, sent_pos_tags, sent_chars in zip(DOC.words, DOC.pos_tags, DOC.chars):
        words_ngram = ngram_split(sent_words, 2)
        pos_tags_ngram = ngram_split(sent_pos_tags, 2)
        char_ngram = ngram_split(sent_chars, 5)
        # doc原始org ngram feature落地
        words_ngram_file.write(' '.join([str(i) for i in words_ngram])), words_ngram_file.write('\n')
        pos_tags_ngram_file.write(' '.join([str(i) for i in pos_tags_ngram])), pos_tags_ngram_file.write('\n')
        chars_ngram_file.write(' '.join([str(i) for i in char_ngram])), chars_ngram_file.write('\n')
        # 添加词典
        count_ngram_tf(words_ngram, word_ngram_count_dict)
        count_ngram_tf(pos_tags_ngram, pos_tag_ngram_count_dict)
        count_ngram_tf(char_ngram, char_ngram_count_dict)
    words_ngram_file.close()
    pos_tags_ngram_file.close()
    chars_ngram_file.close()
    #5、生成dependency_relation_triplets特征
    for dependency_relation_triplet in DOC.dependency_relation_triplets:
        # 原始特征落地
        dependency_relation_triplets_file.write(' '.join(str(i) for i in dependency_relation_triplet))
        dependency_relation_triplets_file.write('\n')
        # 添加词典
        count_ngram_tf(dependency_relation_triplet, dependency_relation_triplet_count_dict)

# 生成文本原始特征和特征词典
def gen_org_feature_and_dicts(basepath):
    dirs = ['org_features/adv', 'org_features/ele', 'org_features/int']
    dir_paths = []
    for dir in dirs:
        dir_path = os.path.join(base_path, dir)
        dir_paths.append(dir_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    path = os.path.join(base_path, 'Texts-SeparatedByReadingLevel')
    file_dirs = os.listdir(path)
    for i, dir in enumerate(file_dirs):
        cur_path = os.path.join(path, dir)
        files = os.listdir(cur_path)
        for file in files:
            file_path = os.path.join(cur_path, file)
            single(file_path, file, dir_paths[i])

    # 生成词典
    n_gram_word_dicts = gen_ngram_dict(word_ngram_count_dict)
    n_gram_pos_tag_dicts = gen_ngram_dict(pos_tag_ngram_count_dict)
    n_gram_char_dicts = gen_ngram_dict(char_ngram_count_dict)
    dependency_relation_triplet_dicts = gen_ngram_dict(dependency_relation_triplet_count_dict)
    dict_path = os.path.join(base_path, 'dicts')
    if not os.path.exists(dict_path):
        os.mkdir(dict_path)
    w = open(dict_path + '/n_gram_word_dicts.txt', 'w')
    p = open(dict_path + '/n_gram_pos_tag_dicts.txt', 'w')
    c = open(dict_path + '/n_gram_char_dicts.txt', 'w')
    d = open(dict_path + '/dependency_relation_triplet_dicts.txt', 'w')
    w.write('\n'.join(str(i) for i in n_gram_word_dicts))
    p.write('\n'.join(str(i) for i in n_gram_pos_tag_dicts))
    c.write('\n'.join(str(i) for i in n_gram_char_dicts))
    d.write('\n'.join(str(i) for i in dependency_relation_triplet_dicts))
    w.close(), p.close(), c.close(), d.close()
    print('=====原始特征路径=====')
    print('{0}\n'.format(i for i in dir_paths) )
    print('=====特征词典路径=====')
    print(dict_path)

if __name__=='__main__':
    '''将Texts-SeparatedByReadingLevel放在data目录下，其它文件将会自动生成
        逻辑：选取n-gram在语料库中出现次数>5的特征作为特征词典，以此特征词典为基础来生成文本特征向量
        注：目前特别是char、dependency的特征向量过于稀疏，可否考虑增大n-gram语料库出现次数？
    '''
    import os
    base_path = r'/Users/yuanlingou/Desktop/Dissertation/Code/OneStopEng_2/data'
    files = os.listdir(base_path + '/Parsed')
    adv_files, ele_files, int_files = [], [], []
    for file in files:
        file_type = file.split('.')[0].split('-')[1]
        if file_type=='adv':
            adv_files.append(file)
        elif(file_type=='ele'):
            ele_files.append(file)
        elif(file_type=='int'):
            int_files.append(file)
        else:
            file_type = file.split('.')[0].split('-')[2]
            if file_type == 'adv':
                adv_files.append(file)
            elif (file_type == 'ele'):
                ele_files.append(file)
            elif (file_type == 'int'):
                int_files.append(file)
    #print(len(adv_files))
    gen_org_feature_and_dicts(base_path)