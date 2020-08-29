from nltk.tree import *

# 特征提取
def extract_single_parse_feature(parse_trees):
    features = {}
    def _ratio_feature(numerator, denominator):
        return round(numerator/denominator, 2)
    # 特征变量定义
    avgParseTreeHeight = 0
    numSubtrees = 0
    numNP = 0
    AvgNPSize = 0
    numVP = 0
    AvgVPSize = 0
    numPP = 0
    AvgPPSize = 0
    numWhPhrases = 0
    reducedRelClauses = 0
    interjections = 0
    numConjPhrases = 0
    numPronouns = 0
    perpronouns = 0
    whperpronouns = 0
    numFunctionWords = 0
    TotalWords = 0
    numVerbs =0
    numVB = 0
    numVBD = 0
    numVBG = 0
    numVBN = 0
    numVBP =0
    numVBZ = 0
    numAdj = 0
    numAdverbs = 0
    numInterjections = 0
    numConjunct = 0
    numNouns = 0
    numProperNouns = 0
    numModals = 0
    numauxverbs = 0
    numDeterminers = 0
    numPrepositions = 0
    TotalSentences = len(parse_trees)
    numLexicals = 0

    # to do
    distHeadWord = 0
    distSemanticHeadWord = 0
    for parse_tree in parse_trees:
        tree = Tree.fromstring(parse_tree)

        avgParseTreeHeight += tree.height()
        subtrees = tree.subtrees()

        for subtree in subtrees:
            # 具有语法结构的节点

            label = subtree.label()

            subtree_num_children = len(subtree) # 子节点个数
            if subtree.height()>2:
                if label=='NP':
                    numNP += 1
                    AvgNPSize += subtree_num_children
                elif(label=='VP'):
                    numVP += 1
                    AvgVPSize += subtree_num_children
                elif(label=='PP'):
                    numPP += 0
                    AvgPPSize += subtree_num_children
                elif(label in ['WHNP', 'WHPP', 'WHAVP', 'WHADJP']):
                    numWhPhrases +=1
                elif(label=='RRC'):
                    reducedRelClauses += 1
                elif(label=='INTJ' or label == 'UH'):
                    interjections += 1
                elif(label=='CONJP'):
                    numConjPhrases += 1
            else: #直连叶子节点
                if label=='PRP' or label=="PRP$" or label=='WP' or label=='WP$':
                    numPronouns += 1
                    if label=="PRP":
                        perpronouns += 1
                    if label=="WP":
                        whperpronouns += 1
                    numFunctionWords += 1
                    TotalWords += 1
                if label in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                    numVerbs += 1
                    TotalWords += 1
                    if label=='VB':
                        numVB += 1
                    elif(label=='VBD'):
                        numVBD += 1
                    elif(label=='VBG'):
                        numVBG += 1
                    elif(label=='VBN'):
                        numVBN += 1
                    elif(label=='VBP'):
                        numVBP += 1
                    elif(label=='VBZ'):
                        numVBZ += 1
                if label in ['JJ', 'JJR', 'JJS']:
                    numAdj  += 1
                    TotalWords += 1
                if label in ['RB', 'RBR', 'RBS', 'RP']:
                    numAdverbs += 1
                    numFunctionWords += 1
                    TotalWords += 1
                if label=='IN':
                    numPrepositions += 1
                    numFunctionWords += 1
                if label == 'UH':
                    numInterjections += 1
                    interjections += 1
                    numFunctionWords +=1
                    TotalWords += 1
                if label == 'CC':
                    numConjunct += 1
                    numFunctionWords +=1
                    TotalWords +=1
                if label == 'NN' or label == 'NNS':
                    numNouns += 1
                    TotalWords +=1
                if label=='NNP' or label =='NNPS':
                    numProperNouns += 1
                    TotalWords += 1
                if label == 'MD':
                    numModals += 1
                    numauxverbs += 1
                    numFunctionWords += 1
                    TotalWords += 1
                if label == 'DT':
                    numFunctionWords += 1
                    numDeterminers += 1
                    TotalWords += 1
        numLexicals += numAdj + numNouns + numVerbs + numAdverbs + numProperNouns

    if TotalWords:
        features['TotalSentences'] = TotalSentences
        features['TotalWords/TotalSentences'] = _ratio_feature(TotalWords, TotalSentences)
        features['numNouns+numProperNouns/TotalWords'] = _ratio_feature(numNouns+numProperNouns, TotalWords)
        features['numProperNouns/TotalWords'] = _ratio_feature(numProperNouns, TotalWords)
        features['numPronouns/TotalWords'] = _ratio_feature(numPronouns, TotalWords)
        features['numConjunct/TotalWords'] = _ratio_feature(numConjunct, TotalWords)
        features['numAdj/TotalWords'] = _ratio_feature(numAdj, TotalWords)
        features['numVerbs/TotalWords'] = _ratio_feature(numVerbs, TotalWords)
        features['numNP/TotalSentences'] = _ratio_feature(numNP, TotalSentences)
        features['numVP/TotalSentences'] = _ratio_feature(numVP, TotalSentences)
        features['numPP/TotalSentences'] = _ratio_feature(numPP, TotalSentences)
        # features['numSBAR'] = _ratio_feature(numSBAR, TotalSentences)
        features['AvgNPSize/numNP'] = _ratio_feature(AvgNPSize, numNP) if numNP else 0
        features['AvgVPSize/numVP'] = _ratio_feature(AvgVPSize, numVP) if numVP else 0
        features['AvgPPSize/numPP'] = _ratio_feature(AvgPPSize, numPP) if numPP else 0
        features['avgParseTreeHeight/TotalSentences'] = _ratio_feature(avgParseTreeHeight, TotalSentences)
        # features['numClauses'] = _ratio_feature(numClauses, TotalSentences)
        # features['TotalWords/numClauses'] = _ratio_feature(TotalWords, numClauses)
        features['numLexicals/TotalWords'] = _ratio_feature(numLexicals, TotalWords)
        features['numAdverbs/numLexicals'] = _ratio_feature(numAdverbs, numLexicals)
        features['numAdj/numLexicals'] = _ratio_feature(numAdj, numLexicals)
        features['numAdj+numAdverbs/numLexicals'] = _ratio_feature(numAdj+numAdverbs, numLexicals)
        features['numNouns+numProperNouns/numLexicals)'] = _ratio_feature(numNouns+numProperNouns, numLexicals)
        features['numVerbs-numauxverbs/numLexicals'] = _ratio_feature(abs(numVerbs-numauxverbs), numLexicals)
        # features['numDependencies'] = _ratio_feature(numDependencies, TotalWords)
        features['distHeadWord/TotalSentences'] =_ratio_feature(distHeadWord, TotalSentences)
        # features['numConstituents'] = _ratio_feature(numConstituents, TotalSentences)
        features['distSemanticHeadWord/TotalSentences'] = _ratio_feature(distSemanticHeadWord, TotalSentences)
        features['numWhPhrases/TotalSentences'] = _ratio_feature(numWhPhrases, TotalSentences)
        features['reducedRelClauses/TotalSentences'] = _ratio_feature(reducedRelClauses, TotalSentences)
        features['interjections/TotalSentences'] = _ratio_feature(interjections, TotalSentences)
        features['numConjPhrases/TotalSentences'] = _ratio_feature(numConjPhrases, TotalSentences)
        features['numAdverbs/TotalSentences'] = _ratio_feature(numAdverbs, TotalSentences)
        features['numModals/TotalSentences'] = _ratio_feature(numModals, TotalSentences)
        # features['numSenses'] = _ratio_feature(numSenses, wordsforwhichsensesarecounted)
        # features['numcommas'] = _ratio_feature(numcommas, TotalSentences)
        features['whperpronouns/TotalSentences'] = _ratio_feature(whperpronouns, TotalSentences)
        features['numFunctionWords/TotalWords'] = _ratio_feature(numFunctionWords, TotalWords)
        features['numDeterminers/TotalWords'] = _ratio_feature(numDeterminers, TotalWords)
        features['numVB/TotalWords'] = _ratio_feature(numVB, TotalWords)
        features['numVBD/TotalWords'] = _ratio_feature(numVBD, TotalWords)
        features['numVBG/TotalWords'] = _ratio_feature(numVBG, TotalWords)
        features['numVBN/TotalWords'] = _ratio_feature(numVBN, TotalWords)
        features['numVBP/TotalWords'] = _ratio_feature(numVBP, TotalWords)
        return features

# 生成特征
def gen_features(files, base_path, type):
    features_dir = base_path + '/paresed_features/' + type
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    features_path = features_dir + '/{}.tsv'.format(type)
    all_features = []
    for file in files:
        file_path = os.path.join(base_path+'/Parsed', file)
        with open(file_path, 'r')as f:
            parse_trees = f.readlines()
        #print(parse_trees)
        single_features = extract_single_parse_feature(parse_trees)
        #print(single_features)
        all_features.append(single_features)
    pd.DataFrame(all_features).to_csv(features_path, sep='\t', index=False)

if __name__=='__main__':
    '''将Parsed文件夹放在data目录下，其它自动生成'''
    import os
    import pandas as pd
    base_path = r'/Users/yuanlingou/Desktop/Dissertation/Code/OneStopEng_2/data'
    files = os.listdir(base_path+'/Parsed')
    adv_files= []
    for file in files:
        file_type = file.split('_')[0]
        if file_type=='Section':
            adv_files.append(file)


    #print(adv_files)
    gen_features(adv_files, base_path, 'Barclays_f')



