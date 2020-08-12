import spacy   
from spacy.matcher import Matcher
from spacy.util import filter_spans
import random


def weighted_random(pairs):
    total = sum(pair[1] for pair in pairs)
    r = random.uniform(0, total)
    for (name, weight) in pairs:
        r -= weight
        if r <= 0: return name

sentences=["The ADJ NOUN ADV TRANVERBs the NOUN.",
"ADJ, ADJ NOUN ADV TRANVERBs a ADJ, ADJ NOUN.",
"NOUN is a ADJ NOUN.",
"INTJ, NOUN!",
"NNS INTRANVERB!",
"The NOUN INTRANVERBs like a ADJ NOUN.",
"NNS INTRANVERB like ADJ NOUN.",
"Why does the NOUN INTRANVERB?",
"INTRANVERB ADV like a ADJ NOUN.",
"NOUN, NOUN, and NOUN.",
"Where is the ADJ NOUN?",
"All NOUNs TRANVERB ADJ, ADJ NOUN.",
"Never TRANVERB a NOUN.",
]
wordtypes=["ADJ","INTRANVERB","TRANVERB","INTJ","ADV","PRPN","VERB","NOUN"]
poem="\n".join(random.sample(sentences,10))
wordlist={}
for wtype in wordtypes:
    #tr4w.analyze(text, candidate_pos = [wtype], window_size=4, lower=False)
    #words+=tr4w.get_keywords(10)
    wordlist[wtype]=words
    while wtype in poem:
        wchoice=weighted_random(wordlist[wtype])
        poem.replace(wtype,wchoice,1)
'''
nlp = spacy.load('en_core_web_sm') 

sentence = 'The cat sat on the mat. He quickly ran to the market. The dog jumped into the water. The author is writing a book.'
pattern = [{'POS': 'VERB', 'OP': '?'},
           {'POS': 'ADV', 'OP': '*'},
           {'POS': 'VERB', 'OP': '+'}]

# instantiate a Matcher instance
matcher = Matcher(nlp.vocab)
matcher.add("Verb phrase", None, pattern)

doc = nlp(sentence) 
# call the matcher to find matches 
matches = matcher(doc)
spans = [doc[start:end] for _, start, end in matches]

print (filter_spans(spans))   '''