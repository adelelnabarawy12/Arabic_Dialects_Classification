import arabic_reshaper                   # pip install arabic-reshaper
from bidi.algorithm import get_display   # pip install python-bidi
from emoji import get_emoji_regexp       # pip install emoji
import nltk
from nltk.stem.isri import ISRIStemmer
import re
import string
import torch
from torch import nn

class RNNModel(nn.Module):
    def __init__(
        self, 
        input_dim,
        embed_size,
        output_dim,
        hidden_size,
        bidirectional,
        num_layers,
        dropout_prop
    ):
        super(RNNModel, self).__init__()
        self.D = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_dim, embed_size)
        self.rnn = nn.GRU(
            embed_size, hidden_size, 
            num_layers=num_layers, 
            bidirectional=bidirectional, 
            batch_first = True
        )
        self.fc = nn.Linear(self.D * hidden_size, output_dim)
        self.dropout = nn.Dropout(dropout_prop)
    # End Func
    
    def forward(self, input):
        N = input.shape[0]
        embedding_out = self.dropout(self.embedding(input))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        h_0 = torch.zeros(self.D * self.num_layers, N, self.hidden_size).to(device)
        _, out = self.rnn(embedding_out, h_0)
        out = torch.cat((out[-2, :, :], out[-1, :, :]), dim = 1) if self.D == 2 else out[-1, :, :] 
        out = self.fc(out)
        return out
    # End Func
    
    def predict(self, input):
        with torch.no_grad():
            return torch.softmax(self.forward(input), dim = 1).argmax(dim = 1)
        # End Block
    # End Func
# End Class

def load_stopwords(file_path = './data/arabic_stopwords_list_nw.txt'):
    with open(file_path, 'r', encoding = 'utf-8') as stopwords_file:
        stopwords = [word.strip() for word in stopwords_file.readlines()]
    # End file stream
    return stopwords
# End Func

def get_label(text):
    cls_names = ['AE', 'BH', 'DZ', 'EG', 'IQ', 'JO', 'KW', 'LB', 'LY', 'MA', 'OM', 'PL', 'QA', 'SA', 'SD', 'SY', 'TN', 'YE']
    model = torch.load("./model_file/the_best_model.pt")
    vocab = torch.load('./model_file/vocab.pt')['vocab']
    tokens = preprocess_text(text, return_tokens = True)
    seq_text = [vocab.stoi[token] for token in tokens]
    return cls_names[model.predict(seq_text).squeeze(0)]
# End Func

def format_arabic(text):
    text = arabic_reshaper.reshape(text)
    text = get_display(text)
    return text
# End Func

def preprocess_text(text, remove_stopwords = False, stem = False, return_tokens = False):
    # Remove mentions, hashtags or any english words and numbers 
    cleaning_regex_script = re.compile(pattern=r'(\@\w+|\#\w+|[A-Za-z0-9]+)')
    text = cleaning_regex_script.sub('', text)

    # Remove emojies
    emoji_regex = get_emoji_regexp()
    text = emoji_regex.sub('', text)

    # Remove punctuations and some symbols
    arabic_punctuations = '''`÷×٪؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ.'''
    symbols = '❤♡❀♩﴾﴿↓❁♬'
    puncts = arabic_punctuations + string.punctuation + symbols
    text = text.translate(str.maketrans('', '', puncts))    

    # Remove Arabic Digits
    arabic_numbers_digits = r'[٠١٢٣٤٥٦٧٨٩]+'
    text = re.sub(arabic_numbers_digits, '', text)

    # Remove unnessary spaces
    spaces_regex_script = re.compile(pattern=r'[\s]{2,}')
    text = spaces_regex_script.sub(' ', text).strip()

    # Remove arabic diacritics
    arabic_diacritics = r'[ًٌٍَُِّـ]'
    text = re.sub(arabic_diacritics, '', text)

    # Normalize the arabic text alpha
    text = re.sub("[إأآ]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)

    # Tokenize text
    tokens = nltk.word_tokenize(text)

    # Remove stop words
    if remove_stopwords:
        stopwords = load_stopwords()
        tokens = [token for token in tokens if token not in stopwords and token.isalpha()]
    # End if
    
    # Get words root using stemming
    if stem:
        stemmer = ISRIStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    # End if
    
    preprocessed_text = tokens if return_tokens else ' '.join(tokens)
    return preprocessed_text
# End Func

def get_label(text):
    cls_names = ['AE', 'BH', 'DZ', 'EG', 'IQ', 'JO', 'KW', 'LB', 'LY', 'MA', 'OM', 'PL', 'QA', 'SA', 'SD', 'SY', 'TN', 'YE']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load("./model_file/the_best_model.pt", map_location=device)
    vocab = torch.load('./model_file/vocab.pt')['vocab']
    tokens = preprocess_text(text, return_tokens = True)
    seq_text = torch.Tensor([int(vocab.stoi[token]) for token in tokens], device = device).unsqueeze(0).type(torch.long)
    return cls_names[model.predict(seq_text).squeeze(0)]
# End Func