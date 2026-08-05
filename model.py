import torch
import torch.nn as nn
import random
class Encoder(nn.Module):
  def __init__(self , vocab_size):
    super().__init__()
    self.emb = nn.Embedding(vocab_size , 128 , padding_idx=0)
    self.dropout = nn.Dropout(0.3)
    self.gru = nn.GRU(128 , 256 , 1 , batch_first=True)

  def forward(self , x):
    emb = self.dropout(self.emb(x))
    output , hidden = self.gru(emb)
    return hidden

class Decoder(nn.Module):
  def __init__(self , vocab_size):
    super().__init__()
    self.emb = nn.Embedding(vocab_size , 128 , padding_idx=0)
    self.dropout = nn.Dropout(0.3)
    self.gru = nn.GRU(128 , 256 , 1 , batch_first=True)
    self.fc = nn.Linear(256 , vocab_size)

  def forward(self , prev_prediction , hidden):
    emb = self.dropout(self.emb(prev_prediction))
    emb = emb.unsqueeze(1)

    output , hidden = self.gru(emb , hidden)
    output = output.squeeze(1)

    prediction = self.fc(output)

    return prediction , hidden


class Seq2Seq(nn.Module):
    def __init__(self , vocab_english , vocab_hindi):
        super().__init__()
        self.encoder = Encoder(len(vocab_english))
        self.decoder = Decoder(len(vocab_hindi))
        self.SOS_IDX = vocab_hindi["<start>"]
        self.EOS_IDX = vocab_hindi["<end>"]
        self.PAD_IDX = vocab_hindi["<PAD>"]

    def forward(self , english_src , hindi_target):
        hidden = self.encoder(english_src)
        decoder_input = hindi_target[: , 0]

        predictions = []

        target_len = hindi_target.shape[1]

        teacher_forcing_ratio = 0.7
        for t in range(1, target_len):

          prediction , hidden = self.decoder(decoder_input , hidden)
          predictions.append(prediction)

          teacher_force = random.random() < teacher_forcing_ratio

          if teacher_force:
            decoder_input = hindi_target[: , t]
          else:
            decoder_input = prediction.argmax(dim=1)

        predictions = torch.stack(predictions, dim=1)
        return predictions

    def predict(self, english_src, max_len=50):

        self.eval()
        with torch.no_grad():

            hidden = self.encoder(english_src)

            decoder_input = torch.tensor([self.SOS_IDX], device=english_src.device)

            predictions = []

            for _ in range(max_len):

                prediction, hidden = self.decoder(decoder_input, hidden)

                next_word = prediction.argmax(dim=1)

                predictions.append(next_word.item())

                if next_word.item() == self.EOS_IDX:
                    break

                decoder_input = next_word

        return predictions

def getModel(vocab_english , vocab_hindi):
    model = Seq2Seq(vocab_english , vocab_hindi)
    return model
    
# "Dataset_English_Hindi.csv"
    