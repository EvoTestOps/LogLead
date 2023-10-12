#from transformers import DistilBertTokenizer, TFDistilBertModel
import tensorflow as tf
from transformers import AlbertTokenizer, TFAlbertModel
import time


class BertEmbeddings:
    def __init__(self):
        
        #Print out all GPU and CPU devices
        devices = tf.config.list_physical_devices()
        print("All physical devices: ", devices)

        gpus = tf.config.list_physical_devices('GPU')
        print("GPUs: ", gpus)

        cpus = tf.config.list_physical_devices('CPU')
        print("CPUs: ", cpus)

        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        self.model = TFAlbertModel.from_pretrained('albert-base-v2', output_hidden_states=True)

        '''
        base bert
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = TFBertModel.from_pretrained('bert-base-uncased')
        DistilBert
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = TFDistilBertModel.from_pretrained('distilbert-base-uncased', output_hidden_states=True)
        '''

    def create_bert_emb(self, sentences):
        #length in word piece tokens
        max_length = 30
        # Set cache batch size depending on GPU memory
        cache_size = 300
        embeddings = []

        # Start tracking time
        start_time = time.time()

        for i in range(0, len(sentences), cache_size):
            batch_sentences = sentences[i: i + cache_size]

            # Tokenize sentences and convert to tensor
            tokenized_batch = self.tokenizer(batch_sentences, truncation=True, padding='max_length',
                                             add_special_tokens=True, return_tensors='tf', max_length=max_length)

            # Put the model in "evaluation" mode, meaning feed-forward operation.
            self.model.trainable = False

            # Predict hidden states features for each layer
            outputs = self.model(tokenized_batch)

            # `outputs` is a tuple with various elements. The hidden states are in the 3rd element.
            hidden_states = outputs['hidden_states']

            # The last layer is at -1 index. Each layer contains embeddings for all tokens.
            token_vecs = hidden_states[-1]

            # Calculate the mean along the sequence length dimension for each sentence
            sentence_embs = tf.reduce_mean(token_vecs, axis=1)

            embeddings.append(sentence_embs.numpy())

        # Concatenate all batch embeddings
        embeddings = tf.concat(embeddings, axis=0)

        # End tracking time
        end_time = time.time()

        # Calculate and print the time taken in seconds
        elapsed_time = end_time - start_time
        print(f"Time taken: {elapsed_time:.2f} seconds")
        return embeddings
