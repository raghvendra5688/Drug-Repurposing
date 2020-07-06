import time
class SmilesDataset:
    '''
    Class to hold a mapping from all possible strings to another,
    and to hold the tokenised values.
    
    Tokenized using a dictionary 
    '''

    def __init__(self, raw_in = None):
        
        self.raw_input = raw_in
        self.n         = 0 # Highest number assigned to tokenmap
        self.batch     = list()
        self.__InvertedTokens__=dict()
        self.Tokenlist = list()
        self.Tokens = dict()
        di = dict()
        atoms = [
                 'Li',
                 'Na',
                 'Al',
                 'Si',
                 'Cl',
                 'Sc',
                 'Zn',
                 'As',
                 'Se',
                 'Br',
                 'Sn',
                 'Te',
                 'Cn',
                 'H',
                 'B',
                 'C',
                 'N',
                 'O',
                 'F',
                 'P',
                 'S',
                 'K',
                 'V',
                 'I',
                ]
        special = [
                   '(', ')', '[', ']', '=', '#', '%', '0', '1', '2', '3', '4', '5',
                   '6', '7', '8', '9', '+', '-', 'se', 'te', 'c', 'n', 'o', 's'
                  ]
        table = ['E','A']+sorted(atoms, key=len, reverse=True) + special
        for counter,symbol in enumerate(table):
            di[symbol] = counter
        self.Tokens, self.Tokenlist, self.n = di, table, len(table)
        self.__InvertedTokens__ = {self.Tokens[x]: \
                                    x for x in self.Tokens}

        if raw_in:
            self.Tokenised = self.tokenise(raw_in)

    def tokenise(self, instring):
        '''
        Read in a string and return the tokenised values
        '''

        N = len(instring)
        i = 0
        seq = list()
        seq.append(self.Tokens["A"])

        timeout = time.time() + 5
        while (i < N):
            for j in range(self.n):
                symbol = self.Tokenlist[j]
                if (symbol==instring[i:i+len(symbol)]):
                    seq.append(self.Tokens[symbol])
                    i+=len(symbol)
                    break;
            if time.time() > timeout:
                break

        seq.append(self.Tokens["E"])
        return seq

    def batch_tokenise(self, x_list):
        '''
        Does tokenisation on a list of strings
        '''

        rvals = list()
        for x in x_list:
            rvals.append(self.tokenise(x))

        self.batch = rvals
        return rvals

    def detokenise(self, seqints):
        '''
        Inverse of tokenise
        '''
        seqlist = list()
        for num in seqints:
            seqlist.append(self.__InvertedTokens__[num])

        return ''.join(seqlist)

def tokenize_drug(text):
    """
    Tokenizes from a string into a list of strings (tokens)
    """
    atoms = ['Li','Na','Al','Si','Cl','Sc','Zn',
             'As','Se','Br','Sn','Te','Cn','H',
             'B','C','N','O','F','P','S','K',
             'V','I'
            ]
    special = ['(', ')', '[', ']', '=', '#', '%', '0', '1', '2', '3', '4', '5',
                   '6', '7', '8', '9', '+', '-', 'se', 'te', 'c', 'n', 'o', 's'
              ]
    table = sorted(atoms, key=len, reverse=True) + special
    N = len(text)
    n = len(table)
    i = 0
    seq = list()
        
    timeout = time.time() + 5
    while (i < N):
        for j in range(n):
            symbol = table[j]
            if (symbol==text[i:i+len(symbol)]):
                seq.append(symbol)
                i+=len(symbol)
                break;
        if time.time() > timeout:
            break

    return seq


def tokenize_protein(text):
    """
    Tokenizes from a proteins string into a list of strings
    """
    aa = ['A','C','D','E','F','G','H','I','K','L',
         'M','N','P','Q','R','S','T','V','W','Y']
    N = len(text)
    n = len(aa)
    i=0
    seq = list()
    
    timeout = time.time()+5
    for i in range(N):
        symbol = text[i]
        if (symbol in aa):
            seq.append(symbol)
        else:
            seq.append('X')
        if time.time() > timeout:
            break
            
    return seq


def test():
    print("\nTesting..........")
    print("Tokenising:\tCOc1cccc\n")
    sd = SmilesDataset("COc1cccc")
    s0 = sd.Tokenised[:]
    print("Tokenised   = %s" % s0)
    s1 = sd.detokenise(s0)
    print("Detokenised = %s\n" % s1)


if __name__=="__main__":
    test()


