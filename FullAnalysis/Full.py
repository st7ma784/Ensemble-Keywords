import csv,os,random
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import OrderedDict,defaultdict
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en_core_web_sm')

def weighted_random(pairs):
    total = sum(pair[1] for pair in pairs)
    r = random.uniform(0, total)
    for (name, weight) in pairs:
        r -= weight
        if r <= 0: return name


class TextRank4Keyword():
    """Extract keywords from text"""
    
    def __init__(self):
        self.d = 0.85 # damping coefficient, usually is .85
        self.min_diff = 1e-5 # convergence threshold
        self.steps = 10 # iteration steps
        self.node_weight = None # save keywords and its weight
        with open('./abstractnounlist.txt','r') as file:
            ABST=file.readlines()
        self.ABSTLIST=[word.lower().replace("\n","") for word in ABST]
        #print(self.ABSTLIST)
    
    def set_stopwords(self, stopwords):  
        """Set stop words"""
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True
    def check_verb(self, token):
        if token.pos_ == 'VERB':
            indirect_object = False
            direct_object = False
            for item in token.children:
                if(item.dep_ == "iobj" or item.dep_ == "pobj"):
                    indirect_object = True
                if (item.dep_ == "dobj" or item.dep_ == "dative"):
                    direct_object = True
            if indirect_object and direct_object:
                return 'TRANVERB'
            elif direct_object and not indirect_object:
                return 'TRANVERB'
            elif not direct_object and not indirect_object:
                return 'INTRANVERB'
            else:
                return 'VERB'
        elif token.pos_ == 'NOUN':
            #print(token.text)
            if token.text.lower() in self.ABSTLIST:
                #print("Found to be ABST" + token.text)
                return 'ABSTNOUN'
            else:
                return token.pos_
        else:
            return token.pos_
    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos == could implement with matcher for more complex structures"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                pos=self.check_verb(token)
                if (pos in candidate_pos or token.tag_ in candidate_pos) and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences
        
    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab
    
    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs
        
    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())
    
    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1
            
        # Get Symmeric matrix
        g = self.symmetrize(g)
        
        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm!=0) # this is ignore the 0 element in norm
        
        return g_norm

    
    def get_keywords(self, number=10):
        """Print top number keywords"""
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        return list(node_weight.items())[:number]

        
        
    def analyze(self, text, 
                candidate_pos=['NOUN', 'PROPN'], 
                window_size=4, lower=False, stopwords=list()):
        """Main function to analyze text"""
        
        # Set stop words
        self.set_stopwords(stopwords)
        
        # Pare text by spaCy
        doc = nlp(text)
        
        # Filter sentences
        sentences = self.sentence_segment(doc, candidate_pos, lower) # list of list of words
        
        # Build vocabulary
        vocab = self.get_vocab(sentences)
        
        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)
        
        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)
        
        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))
        
        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1-self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr))  < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]
        
        self.node_weight = node_weight


def main():
    tr4w = TextRank4Keyword()

    sentences=["The ADJ NOUN ADV TRANVERBs the NOUN.",
    "ADJ, ADJ NOUN ADV TRANVERBs a ADJ, ADJ NOUN.",
    "ABSTNOUN is a ADJ NOUN.",
    "INTJ, ABSTNOUN!",
    "NOUNs INTRANVERB!",
    "The NOUN INTRANVERBs like a ADJ NOUN.",
    "NOUNs INTRANVERB like ADJ NOUN.",
    "Why does the NOUN INTRANVERB?",
    "INTRANVERB ADV like a ADJ NOUN.",
    "ABSTNOUN, ABSTNOUN, and ABSTNOUN.",
    "Where is the ADJ NOUN?",
    "All NOUNs TRANVERB ADJ, ADJ NOUN.",
    "Never TRANVERB a NOUN.",
    ]
    wordtypes=["ADJ","ABSTNOUN","INTRANVERB","TRANVERB","INTJ","ADV","PRPN","VERB","NOUN"]
    poem="\n".join(random.sample(sentences,10))
    wordlist={}
    words=[]
    Filename=os.path.join("DATA","DataoftheHeart_LakeDistrictsurvey.csv")
    totalsentiment=defaultdict(int)
    scores=list()
    totals=defaultdict(int)
    sid=SentimentIntensityAnalyzer()
    with open(Filename, mode='r') as infile:
        reader = list(csv.DictReader(infile))
        desiredkey=list(reader[0].keys())[17]
        for row in reader:
            text=row.get(desiredkey,"N/A")
            scores.append(sid.polarity_scores(text))
        for wtype in wordtypes:
            print("Now looking for common "+ wtype)
            for row in reader:
                
                text=row.get(desiredkey,"N/A")
                tr4w.analyze(text, candidate_pos = [wtype], window_size=4, lower=False)
                wordlist[wtype]=wordlist.get(wtype,list())+tr4w.get_keywords(10)
            while wtype in poem:
                wchoice=weighted_random(wordlist[wtype])
                if wchoice is None:
                    poem=poem.replace(wtype,"Oh",1) #because we've found a place for interjections. GRR
                else:
                    poem=poem.replace(wtype,wchoice,1)
    for word,score in wordlist['ADJ']:
        totals[word]=totals.get(word,0)+score
    totals=OrderedDict(sorted(totals.items(), key=lambda t: t[1], reverse=True))
    for score in scores:
        for key,value in score.items():
            totalsentiment[key] += value
    for key in totalsentiment:
        totalsentiment[key]=totalsentiment[key]/len(scores)
    print(list(totals.items())[:10])
    print(totalsentiment)
    print("From: "+desiredkey)
    print(poem)
if __name__=="__main__":
    main()
