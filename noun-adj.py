import spacy
from spacy.symbols import NOUN, ADJ

parser = spacy.load("en_core_web_md")
sentence = parser("i want a red chair which is blue and has color green")

# Finding a verb with a subject from below — good
adj_noun_str = ""
for possible_adj in sentence:
    if possible_adj.pos == ADJ:
        ancestor = possible_adj.head
        while (ancestor.dep_ != "ROOT"):
            if ancestor.pos == NOUN:
                break
            ancestor = ancestor.head
        if ancestor.pos == NOUN:
            adj_noun_str += " " + possible_adj.text + " " + ancestor.text
        else:
            adj_noun_str += " " + possible_adj.text
print(adj_noun_str)
