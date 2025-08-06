import torch

class BigramModel:
    def __init__(self,data):
        self.data = data
        self.vocab = sorted(list(set(''.join(self.data)))) # first we create a list of unique characters
        self.s_to_i = { s : i+1 for i,s in enumerate(self.vocab)} # create a mapping from character to index
        self.s_to_i['#']=0 # add a special character for padding
        self.i_to_s = { i : s for s,i in self.s_to_i.items()} # create a mapping from index to character
        self.no_distinct_chars = len(self.s_to_i) #including the padding character
        self.N = torch.zeros((self.no_distinct_chars, self.no_distinct_chars),dtype=torch.int32)

    def build_bigram(self):

        for word in self.data:
            chs = ['#'] + list(word) + ['#'] # add padding character at the beginning and end
            for c1,c2 in zip(chs , chs[1:]):
                ix_1 = self.s_to_i[c1] # get the index of the first character
                ix_2 = self.s_to_i[c2] # get the index of the second character
                self.N[ix_1, ix_2] += 1 # increment the count of

                # the bigram (c1,c2) in the N matrix
    def get_prob_matrix(self, reg_const=1):
        self.P = (self.N + reg_const).float()
        self.P /= self.P.sum(1, keepdim=True)
        return self.P
    
    def sample(self, num=10, seed=23651):
        g = torch.Generator().manual_seed(seed)
        self.get_prob_matrix()
        for _ in range(num):
            index=0
            chr_str=""
            while True:
                p_row = self.P[index]
                #Sample from the distribution
                index = torch.multinomial(p_row, num_samples=1,replacement=True, generator=g).item()  # sample one index from the distribution
                curr_chr= self.i_to_s[index]  # get the character corresponding to the index
                chr_str += curr_chr  # append the character to the string
                if index == 0:  # if the index is 0, it means we have reached the end of the string
                    break
            print(chr_str)  # print the generated string

    def evaluate(self, data=None, print_data=False):
        """ Evaluate the model on a given list of words (or on training data if none is given).
        Prints per-bigram probability, log-likelihood, and total average NLL.
        """
        if data is None:
            data = self.data  # default to full training data
    
        self.get_prob_matrix()
        # Evaluate the quality of the model
        log_likelihood = 0.0
        n=0
        for w in data:
            chs = ['#'] + list(w) + ['#']
            for ch1, ch2 in zip(chs,chs[1:]):
                ix1= self.s_to_i[ch1]
                ix2= self.s_to_i[ch2]
                prob = self.P[ix1, ix2]  # get the probability of the next character given the previous one
                log_prob = torch.log(prob)  # take the log of the probability
                log_likelihood += log_prob  # accumulate the log probability
                n+=1
                if print_data:
                    # Print the bigram and its probability
                    print(f'{ch1}{ch2} -> {prob:.4f} ({log_prob:.4f})')  # print the character pair and its probability
        print(f'Log likelihood: {log_likelihood.item()}')  # print the log likelihood of the model
        nll=-log_likelihood/n  # negative log likelihood
        print(f'Negative log likelihood: {nll.item()}')  # print the negative log likelihood
        