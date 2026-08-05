import torch
from torch.utils.data import Dataset , DataLoader
import pandas as pd

class TranslationDataset(Dataset):
  def __init__(self , df):
    self.x = df["English"].tolist()
    self.y = df["Hindi"].tolist()

  def __len__(self):
    return len(self.x)

  def __getitem__(self , index):
    return torch.tensor(self.x[index] , dtype=torch.long) , torch.tensor(self.y[index] , dtype=torch.long)

def get_data(path):
    df = pd.read_csv(path)


    df["English"] = df["English"].str.lower().str.split()
    df["Hindi"] = df["Hindi"].str.split()

    df["Hindi"] = df["Hindi"].apply(
        lambda x: ["<start>"] + x + ["<end>"] if isinstance(x, list) else ["<start>", "<end>"]
        )

    df_subset = df.head(10000)


    df_subset = df_subset.dropna(subset=['English', 'Hindi'])

    vocab_english = []
    for i , sentance in enumerate(df_subset["English"]) :
      if isinstance(sentance, list):
        vocab_english.extend(sentance)

    vocab_english = list(set(vocab_english))
    vocab_english = ["<PAD>" , "<UNK>"] + vocab_english

    vocab_hindi = []
    for i , sentance in enumerate(df_subset["Hindi"]) :
      vocab_hindi.extend(sentance)

    vocab_hindi = list(set(vocab_hindi))
    # remove special tokens from set so they're not duplicated when we prepend them
    vocab_hindi = [w for w in vocab_hindi if w not in ("<PAD>" , "<start>" , "<end>")]
    vocab_hindi = ["<PAD>" , "<start>" , "<end>"] + vocab_hindi

    eng_word_2_idx = {word : idx for idx , word in enumerate(vocab_english)}
    hindi_word_2_idx = {word : idx for idx , word in enumerate(vocab_hindi)}

    def word_to_number(sentance , vocab):
      value = []
      for i in sentance:
        value.append(vocab.get(i , vocab.get("<UNK>" , 0)))
      return value

    temp_for_eng = []
    temp_for_hindi = []
    for i , j in zip(df_subset["English"] , df_subset["Hindi"]):
      temp_for_eng.append(word_to_number(i , eng_word_2_idx))
      temp_for_hindi.append(word_to_number(j , hindi_word_2_idx))

    df_subset["English"] = temp_for_eng
    df_subset["Hindi"] = temp_for_hindi

    train_dataset = TranslationDataset(df_subset)
    return train_dataset , eng_word_2_idx , hindi_word_2_idx , word_to_number


def word_to_number(sentance , vocab):
    value = []
    for i in sentance:
        value.append(vocab.get(i , vocab.get("<UNK>" , 0)))
    return value

def num_to_word(indices , idx_to_word):
    value = []
    for i in indices:
        word = idx_to_word.get(i , "<UNK>")
        if word == "<end>":
            break
        if word not in ("<PAD>" , "<start>"):
            value.append(word)
    return " ".join(value)