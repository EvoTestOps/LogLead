import tensorflow as tf
from transformers import AlbertTokenizer, TFAlbertModel
from transformers import BertTokenizer, TFBertModel
import time


class BertEmbeddings:
    def __init__(self, bertmodel = "basebert"):

        self.basebert = bertmodel
        
        #Print out all GPU and CPU devices
        devices = tf.config.list_physical_devices()
        print("All physical devices: ", devices)

        gpus = tf.config.list_physical_devices('GPU')
        print("GPUs: ", gpus)

        cpus = tf.config.list_physical_devices('CPU')
        print("CPUs: ", cpus)

        if self.basebert == 'basebert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = TFBertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
            print("Using basebert")

        if self.basebert == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            self.model = TFAlbertModel.from_pretrained('albert-base-v2', output_hidden_states=True)
            print("Using albert")

    def create_bert_emb(self, sentences):
        #length in word piece tokens
        max_length = 30
        # Set cache batch size depending on GPU memory
        cache_size = 1000
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

            inputs = {'input_ids': tokenized_batch['input_ids'],
                      'attention_mask': tokenized_batch['attention_mask']}
            outputs = self.model(inputs)

            if self.basebert == 'basebert':
                hidden_states = outputs[2]
                token_vecs = hidden_states[-1]

            if self.basebert == 'albert':
                hidden_states = outputs['hidden_states']
                # The last layer is at -1 index. Each layer contains embeddings for all tokens.
                token_vecs = hidden_states[-1]

            # Calculate the mean along the sequence length dimension for each sentence
            sentence_embs = tf.reduce_mean(token_vecs, axis=1)

            embeddings.append(sentence_embs)

        # Concatenate all batch embeddings
        embeddings = tf.concat(embeddings, axis=0)

        # End tracking time
        end_time = time.time()

        # Calculate and print the time taken in seconds
        elapsed_time = end_time - start_time
        print(f"Time taken: {elapsed_time:.2f} seconds")
        return embeddings
