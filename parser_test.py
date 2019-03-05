# parser test 

# import nltk 
# from nltk.parse.stanford import StanfordDependencyParser

import spacy
import json

def main():
    nlp = spacy.load('en')
    sents = nlp(u'A woman is walking through the door.')
    print(type(sents))
    print(sents)

    for token in sents: 
        print("token: ", token)
        print("POS: ", token.pos_)
        print("DEPS: ", token.dep_)
        # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
  #         token.shape_, token.is_alpha, token.is_stop)
    all_words = {}
    with open('./data/train-v2.0_madhu.json') as json_file:
        file = json.load(json_file)
        print("k")
        for line in file['data']:
            # print(line)
            paras = line['paragraphs']
            for para in paras:
                qas = para['qas']
                questions = []
                for qa in qas: 
                    question = qa['question']
                    questions.append(question)
                    # print(question)

                context = para['context']
                # print("question: ", question)
                # print("context: ", context)



            break
    # x = {}
    # x['1'] = 2

    # with open('x.json', 'w') as outfile:
    #     json.dump(x, outfile)

    with open('./data/word2idx_madhu.json') as json_file:
        wordidx2pos = {}
        word2idx = json.load(json_file)
        pos2idx = {}
        counter = 0
        pos_counter = 0
        for word in word2idx.keys():
            thing = nlp(word)
            for token in thing: 
                # print("new tokne: ", token.pos_)
                pos = token.pos_
                idx = word2idx[word]
                wordidx2pos[idx] = pos
                if pos not in pos2idx.keys():
                    pos2idx[pos] = pos_counter
                    pos_counter+=1
                print(counter)
                counter+=1
        # print(x)
        print(wordidx2pos)

    wordidx2posidx = {}
    for key in wordidx2pos.keys():
        wordidx2posidx[key] = pos2idx[wordidx2pos[key]]


    with open('wordidx2posidx.json', 'w') as outfile:
        json.dump(wordidx2posidx, outfile)







if __name__ == '__main__':
    main()