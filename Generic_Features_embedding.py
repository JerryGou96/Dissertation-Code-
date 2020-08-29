import os
dict_files = {
    'char': 'n_gram_char_dicts.txt',
    'pos_tag': 'n_gram_pos_tag_dicts.txt',
    'word': 'n_gram_word_dicts.txt',
    'dependency': 'dependency_relation_triplet_dicts.txt'
}

# 加载词典
def load_dicts(feature_type, dicts_dir_path):
    path = os.path.join(dicts_dir_path, dict_files[feature_type])
    with open(path, 'r')as f:
         dicts = ''.join(f.readlines()).split('\n')
    res = {}
    for i, item in enumerate(dicts):
        res[item.replace(', ', ',')] = i+1
    return res, len(res)

# 特征生成
def feature2embedding(file_path, dicts):
    file_features = []
    with open(file_path, 'r')as f:
        features = ''.join(f.readlines()).split('\n')
        tmp_features = {}
        for feature in features:
            feature = feature.replace(', ', ',')
            for item in feature.split(' '):
                if item not in tmp_features.keys():
                    tmp_features[item] = 1
                else:
                    tmp_features[item] +=1
    # 求文本特征和词典的交集
    ret = [i for i in tmp_features.keys() if i in dicts.keys()]
    # 生成特征
    for k in dicts.keys():
        if k not in ret:
            file_features.append(0)
        else:
            file_features.append(tmp_features[k])
    return file_features

# 原始特征转成特征向量
def get_org_features_file(base_path, dif_level):
    def _del_file(file):
        if os.path.exists(file):
            os.remove(file)
    dif_level_path = base_path + '/org_features/' + dif_level
    des_feature_path = base_path + '/features/' + dif_level
    if not os.path.exists(des_feature_path):
        os.makedirs(des_feature_path)
    dicts_path = base_path + '/dicts'
    file_list = os.listdir(dif_level_path)
    char_files, pos_tag_files, word_files, dependency_files = [],[],[],[]
    for file in file_list:
        file_split = file.split('-')
        if 'char' in file_split:
            char_files.append(file)
        elif('pos' in  file_split):
            pos_tag_files.append(file)
        elif('word' in file_split):
            word_files.append(file)
        elif('dependency' in file_split):
            dependency_files.append(file)
    assert len(char_files)==len(pos_tag_files)==len(word_files)==len(dependency_files)
    char_dicts, _ = load_dicts('char', dicts_path)
    pos_tag_dicts, _ = load_dicts('pos_tag', dicts_path)
    word_dicts, _ = load_dicts('word', dicts_path)
    dependency_dicts, _ = load_dicts('dependency', dicts_path)
    _del_file(des_feature_path+'/char.txt')
    _del_file(des_feature_path + '/word.txt')
    _del_file(des_feature_path + '/pos_tag.txt')
    _del_file(des_feature_path + '/dependency.txt')
    char_features_embedding = open(des_feature_path+'/char.txt', 'a')
    word_features_embedding = open(des_feature_path + '/word.txt', 'a')
    pos_tag_features_embedding = open(des_feature_path + '/pos_tag.txt', 'a')
    dependency_features_embedding = open(des_feature_path + '/dependency.txt', 'a')

    for char_file, pos_tag_file, word_file, dependency_file in zip(char_files, pos_tag_files, word_files, dependency_files):
        char_file_path = os.path.join(dif_level_path, char_file)
        pos_tag_file_path = os.path.join(dif_level_path, pos_tag_file)
        word_file_path = os.path.join(dif_level_path, word_file)
        dependency_file_path = os.path.join(dif_level_path, dependency_file)
        tied_lst = [(char_file_path, char_dicts, char_features_embedding), (pos_tag_file_path, pos_tag_dicts, pos_tag_features_embedding),
                    (word_file_path, word_dicts, word_features_embedding), (dependency_file_path, dependency_dicts, dependency_features_embedding)]
        for file_feature_path, dicts, file in tied_lst:
            file_feature_embedding = feature2embedding(file_feature_path, dicts)
            file.write(' '.join([str(i) for i in file_feature_embedding]))
            file.write('\t{0}\n'.format(dif_level))

    char_features_embedding.close()
    word_features_embedding.close()
    pos_tag_features_embedding.close()
    dependency_features_embedding.close()


if __name__=='__main__':
    # 自动生成所有的embedding特征
    base_path = r'D:\private\QA\OneStopEng\data'
    get_org_features_file(base_path, 'adv')
    get_org_features_file(base_path, 'ele')
    get_org_features_file(base_path, 'int')