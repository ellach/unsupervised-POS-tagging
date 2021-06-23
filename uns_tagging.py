from collections import Counter
import random
import itertools
import numpy as np


PENN_TAGS = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP',
             'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG',
             'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

UNIVERSAL_TAGS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT',
                  'SCONJ', 'SYM', 'VERB', 'X']


class POSTags:
    def __init__(self, filePath):
        """
        parses file
        :param filePath: file path
        """
        self.tags, self.sentences = [], []
        with open(filePath, mode='r', encoding='utf-8-sig') as file:
            cur_sentence = []
            for line in file.readlines():
                ls = line.split()
                if len(line) == 1:
                    self.sentences.append(cur_sentence)
                    cur_sentence = []
                else:
                    self.sentences.append(ls)

        self.vocab = set([item for sublist in self.sentences for item in sublist])

    def e_calc(self, words_and_tags, word, tag):

        """
        Emission probability
        e = p(w_k|t_k) = p(book|NN)= count(book, NN)/ count(NN) - MLE
        """

        tags = [tag for (word, tag) in words_and_tags]
        tags_counter = Counter(tags)
        words_and_tags_counter = Counter(words_and_tags)
        return words_and_tags_counter[(word, tag)] / tags_counter[tag]

    def q_calc(self, words_and_tags, next_tag, cur_tag):
        """
        Transition probability
        q = p(t_k+1|t_k) = count(t_k, t_k+1)/count(t_k) -  MLE
        q = p(c|a, b) = lambda1*count(a, b, c)/count(a, b) + lambda2*count(b, c)/count(b) + lambda3*count(c)/num of words
        lambda1 + lambda2 + lambda3 = 1
        """

        lamda1, lamda2 = 0.6, 0.4
        tags = [tag for (word, tag) in words_and_tags]
        paired_tags = Counter(itertools.zip_longest(tags, tags[2:]))
        single_tags = Counter(tags)
        if paired_tags[(next_tag, cur_tag)] > 0:
            paired_tags[(next_tag, cur_tag)] -= 1
        if single_tags[next_tag] > 0:
            single_tags[next_tag] -= 1

        return lamda1 * paired_tags[(next_tag, cur_tag)] / single_tags[cur_tag] + lamda2 * single_tags[cur_tag] / len(
            self.vocab)

    def tag_text(self, random_tagged_text):
        for sentence in self.sentences:
            for word in sentence:
                if not word in random_tagged_text:
                    random_tagged_text.append((word, random.choice(UNIVERSAL_TAGS)))
        return random_tagged_text

    def gibbs_sampling(self, iterations):
        tagged_sentences = []
        tagged_sentences = self.tag_text(tagged_sentences)
        self.tags = [tag for (_, tag) in tagged_sentences]
        words = [word for (word, _) in tagged_sentences]
        results = {}
        print("Tags:  ", self.tags)
        for _ in range(iterations):
            for i, (word, tag) in enumerate(tagged_sentences):
                results[i] = []
                probabilities = [0] * len(self.tags)
                for j, cur_tag in enumerate(self.tags):
                    q1, q2 = 0.0, 0.0
                    if j < len(self.tags) - 1:
                        q1 = self.q_calc(tagged_sentences, self.tags[j + 1], cur_tag)
                    if j > 0:
                        q2 = self.q_calc(tagged_sentences, cur_tag, self.tags[j - 1])
                    e = self.e_calc(tagged_sentences, word, cur_tag)
                    probabilities[j] = e * q1 * q2
                prob = [i for i, p in enumerate(probabilities) if p > 0]
                results[i].append(prob)
            self.tags = self.update_tags(results)
        a = self.get_results(results, words, self.tags)

        with open("output.txt", 'w') as output_file:
            for element, tag in zip(a, self.tags):
                output_file.write("Tag: " + tag +"   After train:   "+str(element[0]) + " , " + str(element[1]) + '\n')
        return a

    def update_tags(self, probs):
        new_tags = [0] * len(self.tags)
        for (i, probs_of_word) in probs.items():
            for probs in probs_of_word:
                index = random.choice(probs)
                new_tags[i] = self.tags[index]
        return new_tags

    def get_results(self, probs, words, tags):
        results = [0] * len(tags)
        for i, word in enumerate(words):
            index = np.argmax(probs[i])
            results[i] = (word, tags[index])
        return results


if __name__ == '__main__':
    pt = POSTags('/Users/ellaeidlin/PycharmProjects/GibbsSampling/sample.txt')
    pt.gibbs_sampling(1000)


"""
Universal POS tags

ADJ: adjective - big, old, green 
ADP: adposition - in, to, during
ADV: adverb - very, well, exactly
AUX: auxiliary - has (done), is (doing), will (do), should (do), must (do)
CCONJ: coordinating conjunction - and, or, but
DET: determiner - a, an, the, 
INTJ: interjection - psst, bravo, hello
NOUN:  noun - girl, cat, tree 
NUM: numeral - 1, 2, 3, one, two, three 
PART: particle - not, ‘s, nicht
PRON: pronoun - I, you, he, she, it, we, they, myself, yourself, himself, herself, itself, ourselves, yourselves, theirselves
PROPN: proper noun - Mary, John, London, NATO, HBO
PUNCT: punctuation - ., :, ?
SCONJ: subordinating conjunction - that, if, while
SYM: symbol - $, %, §, ©
VERB: verb - run, eat
X: other - xfgh pdl jklw

"""
"""
Penn Treebank tagset -  proper for English

CC	Coordinating conjunction - and,but,or...
CD	Cardinal number
DT	Determiner
EX	Existential - there 
FW	Foreign word
IN	Preposition or subordinating conjunction
JJ	Adjective
JJR	Adjective, comparative
JJS	Adjective, superlative
LS	List item marker
MD	Modal - can, could, might, may...
NN	Noun, singular or mass
NNS	Noun, plural
NNP	Proper noun, singular
NNPS	Proper noun, plural
PDT	Predeterminer
POS	Possessive ending
PRP	Personal pronoun -  I, me, you, he...
PRP$	Possessive pronoun - my, your, mine, yours...
RB	Adverb
RBR	Adverb, comparative
RBS	Adverb, superlative
RP	Particle
SYM	Symbol
TO	to
UH	Interjection - uh, well, yes, my...
VB	Verb, base form
VBD	Verb, past tense
VBG	Verb, gerund or present participle
VBN	Verb, past participle
VBP	Verb, non-3rd person singular present
VBZ	Verb, 3rd person singular present
WDT	Wh-determiner
WP	Wh-pronoun - what, who, whom...
WP$	Possessive wh-pronoun -  whose
WRB	Wh-adverb -  how, where why

"""