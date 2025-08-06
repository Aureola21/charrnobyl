import torch
import torch.nn.functional as F

#NEURAL NETWORK MODEL 
# We will use a neural network to predict the next character given the previous one.

class NeuralNetModel:
    def __init__(self,data,seed=23651):
        self.data = data
        self.vocab = sorted(list(set(''.join(self.data))))
        self.s_to_i = {s: i+1 for i, s in enumerate(self.vocab)}
        self.s_to_i['#'] = 0  # Start/end token
        self.i_to_s = {i: s for s, i in self.s_to_i.items()}
        self.no_distinct_chars = len(self.s_to_i)
        #Initializing the network parameters
        self.g = torch.Generator().manual_seed(seed)
        self.W = torch.randn((self.no_distinct_chars, self.no_distinct_chars), generator=self.g, requires_grad=True)
        self.losses=[]

    def create_train_data(self):
        '''Create training data from the input data.
        returns x_train, y_train where 
        x_train contains the input characters and y_train contains the target characters.
        '''
        x_train, y_train = [], []

        for w in self.data:
            chs=['#'] + list(w) + ['#']           
            for ch1,ch2 in zip(chs, chs[1:]):
                ix1= self.s_to_i[ch1] 
                ix2= self.s_to_i[ch2]
                x_train.append(ix1)
                y_train.append(ix2)

        x_train = torch.tensor(x_train)
        y_train = torch.tensor(y_train)
        num = x_train.nelement()
        print('Number of examples: ', num)
        print('-----------------------------------')
        return x_train, y_train
    
    def train(self, x_train, y_train, epochs=1000, lr=0.01,reg_lambda=0.01):
        '''Train the neural network model.
        x_train: input characters
        y_train: target characters
        epochs: number of epochs to train
        lr: learning rate
        '''

        x_train, y_train = self.create_train_data()
        print(f'Training the model for {epochs} epochs with learning rate {lr} and regularization lambda {reg_lambda}')
        print('-----------------------------------')
        for epoch in range(epochs):
            #forward pass:
            x_enc= F.one_hot(x_train, num_classes=self.no_distinct_chars).float() #input to the network : one-hot encoding
            logits = x_enc @ self.W # log-counts
            counts = logits.exp() #Equivalent N
            prob = counts / counts.sum(1, keepdim=True)  # normalize to make it a probability distribution (softmax)
            loss = -prob[torch.arange(x_enc.shape[0]), y_train].log().mean() # negative log likelihood loss
            loss += reg_lambda*(self.W**2).mean() #Regularization Loss
            self.losses.append(loss.item())
            # backward pass
            self.W.grad= None  # reset the gradients
            loss.backward()

            # update the weights
            self.W.data += -lr * self.W.grad  # gradient descent step

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    def sample(self, num=10, seed=23651):
        '''Sample from the neural network model.
        num: number of samples to generate
        seed: random seed for reproducibility
        '''
        g = torch.Generator().manual_seed(seed)
        for _ in range(num):
            out = []
            ix = 0
            while True:
                x_enc = F.one_hot(torch.tensor([ix]), num_classes=self.no_distinct_chars).float()
                logits = x_enc @ self.W
                counts = logits.exp()
                prob = counts / counts.sum(1, keepdim=True)

                ix = torch.multinomial(prob, num_samples=1, replacement=True, generator=g).item()
                if ix == 0:
                    break
                out.append(self.i_to_s[ix])
            print(''.join(out))
    
    def plot_train_loss(self):
        '''Plot the training loss over epochs.'''
        import matplotlib.pyplot as plt
        plt.plot(self.losses)
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid()
        plt.show()
        
    def evaluate(self, data=None,print_data=False):
        """
        Evaluate the model on a given list of words (or on training data if none is given).
        Prints per-bigram probability, log-likelihood, and total average NLL.
        """
        if data is None:
            data = self.data  # default to full training data

        x_eval, y_eval = [], []

        for w in data:
            chs = ['#'] + list(w) + ['#']
            for ch1, ch2 in zip(chs, chs[1:]):
                x_eval.append(self.s_to_i[ch1])
                y_eval.append(self.s_to_i[ch2])

        x_eval = torch.tensor(x_eval)
        y_eval = torch.tensor(y_eval)

        x_enc= F.one_hot(x_eval, num_classes=self.no_distinct_chars).float() #input to the network : one-hot encoding
        logits = x_enc @ self.W # log-counts
        counts = logits.exp() #Equivalent N
        P = counts / counts.sum(1, keepdim=True)  # normalize to make it a probability distribution (softmax)
        nlls = torch.zeros(x_eval.shape[0])

        for i in range(x_eval.shape[0]):
            x = x_eval[i].item()
            y = y_eval[i].item()
            prob = P[i, y]
            logp = torch.log(prob)

            if print_data:
                print(f"Evaluating Bigram {i + 1}/{x_eval.shape[0]}")
                print("---------------")
                print(f"Bigram: {self.i_to_s[x]}{self.i_to_s[y]}")
                print(f"→ Probability: {prob.item():.4f}")
                print(f"→ Log Likelihood: {logp.item():.4f}")
                print(f"→ Negative Log Likelihood: {-logp.item():.4f}")
            nlls[i] = -logp

        print("======================================")
        print(f"📊 Average Negative Log Likelihood: {nlls.mean().item():.4f}")
    
