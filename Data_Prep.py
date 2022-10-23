import re
import nltk
import numpy as np
# nltk.download('punkt')
from tqdm import tqdm

class DataPrep:
    '''
    Prepare corpus before feeding into the NET
    ...
    Attributes
    ----------
    freq_th : Number of occurance for a token in order to consider in the vocabulary 
    ''' 
    def __init__(self, filepath=None, start_line=None, end_line=None, freq_th=1):
        self.filepath = filepath
        self.freq_th = freq_th
        self.start_line = start_line
        self.end_line = end_line
        self.idxtos = {0: '<UNK>'}
        self.stoidx = None
    
    def build_dictionary(self, corpus=None):
        """
        Create a  dictionary {token: index} for all train data unique tokens
        """
        self.corpus = self._read_corpus() if (self.filepath and not corpus) else corpus
        tokens = nltk.tokenize.word_tokenize(' '.join(self.corpus))
        freq = nltk.FreqDist(tokens)
        self.idxtos.update({i+len(self.idxtos): j.lower() \
                            for i,(j,k) in enumerate(dict(freq).items()) \
                            if (k >= self.freq_th and not re.findall('\d+', j.lower()))})
        self.stoidx = dict(zip(self.idxtos.values(), self.idxtos.keys()))
        self.vocab_size = len(self.stoidx) + 1
        
    def _read_corpus(self):
        lines = []
        with open(self.filepath, 'r') as f:
            for pos, line in enumerate(f):
                if (self.start_line < pos < self.end_line):
                    if len(line.strip()) > 0:
                        lines.append(line)

        corpus = " ".join(lines)
        corpus = nltk.tokenize.word_tokenize(corpus)
        corpus =[
            x for x in list(map(lambda x: x.strip().lower(), corpus)) if len(x) > 1
        ]

        return corpus

    
    def _W_to_idx(self, w):
        """
        Returns the Word index in the Vocabulary
        """
        return self.stoidx[w.lower()] if w.lower() in self.stoidx else self.stoidx['<UNK>']
    
    def Decode(self, words):
        if not type(words) == list : words = [words]
        vects = []
        for w in words:
            widx = self._W_to_idx(w)
            vect = np.zeros(self.vocab_size)
            vect[widx] = 1
            vects.append(torch.tensor(vect))
        return torch.cat(vects).flatten()