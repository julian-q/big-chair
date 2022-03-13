import spacy
from spacy.symbols import NOUN, ADJ, ROOT

parser = spacy.load("en_core_web_sm")
sentence = parser("I want a red chair which is blue and has color yellow.")

# Finding a verb with a subject from below — good
adj_noun_list = []
for possible_adj in sentence:
    if possible_adj.text == "yellow":
        print(possible_adj.head.head.head.text)
    if possible_adj.pos == ADJ:
        if possible_adj.head.pos == NOUN:
            adj_noun_list.append(possible_adj.text + " " + possible_adj.head.text)
        elif possible_adj.head.head.pos == NOUN:
            adj_noun_list.append(" " + possible_adj.text + " " + possible_adj.head.head.text)
        else:
            adj_noun_list.append(" " + possible_adj.text)
print(adj_noun_list)
