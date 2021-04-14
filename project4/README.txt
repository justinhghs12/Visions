I had difficulty with the python 2.7 versions of this homework so I found python 3 versions and they are referenced below. 
Something to note is the Meteor values are not recorded. I tried to find the reason why it might not be working or a way to solve
the issue but couldn't find anything. I'm not sure if it makes a difference but thought it was worth noting. 


Python 3 CocoAPI:

Philferriere. “Philferriere/Cocoapi.” GitHub, 24 Oct. 2018, github.com/philferriere/cocoapi. 

Python 3 pycocoevalcap:

Salaniz. “Salaniz/Pycocoevalcap.” GitHub, 28 Mar. 2020, github.com/salaniz/pycocoevalcap.

Using nn.LSTM:

computing Bleu score...
{'testlen': 988, 'reflen': 980, 'guess': [988, 888, 788, 688], 'correct': [628, 285, 110, 45]}
ratio: 1.0081632653050938
Bleu_1: 0.636
Bleu_2: 0.452
Bleu_3: 0.305
Bleu_4: 0.208
computing Rouge score...
ROUGE_L: 0.471
computing CIDEr score...
CIDEr: 0.685


Using my implementation:

computing Bleu score...
{'testlen': 986, 'reflen': 975, 'guess': [986, 886, 786, 686], 'correct': [667, 296, 110, 43]}
ratio: 1.0112820512810141
Bleu_1: 0.676
Bleu_2: 0.475
Bleu_3: 0.316
Bleu_4: 0.211
computing Rouge score...
ROUGE_L: 0.481
computing CIDEr score...
CIDEr: 0.734
