import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader , Dataset
from model import getModel
from dataset import get_data
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)
train_dataset , vocab_english , vocab_hindi , _ = get_data("Dataset_English_Hindi.csv")


def collate_fn(batch):
    PAD_IDX = 0
    src = [x[0] for x in batch]
    trg = [x[1] for x in batch]

    src = pad_sequence(
        src,
        batch_first=True,
        padding_value=PAD_IDX
    )
    trg = pad_sequence(
        trg,
        batch_first=True,
        padding_value=PAD_IDX
    )
    return src, trg

if __name__ == '__main__':
    dataLoader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=False
    )

    model = getModel(vocab_english , vocab_hindi)
    model = model.to(device)

    epochs = 10
    learning_rate = 0.001


    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(params=model.parameters() , lr=learning_rate)

    losses = []
    print("training model")
    for epoch in range(epochs):
      total_loss = 0
      for i , (src , target) in enumerate(dataLoader):
        optimizer.zero_grad()
        # Move tensors to device asynchronously
        src = src.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(src , target)

        loss = loss_fn(output.reshape(-1, len(vocab_hindi)),target[:, 1:].reshape(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        if device.type == "mps":
            torch.mps.empty_cache()
        if i % 250 == 0:  # Print progress more frequently with larger batch sizes
          print(f"  Epoch {epoch+1} | Batch {i}/{len(dataLoader)} | Loss: {loss.item():.4f}")

      avg_loss = total_loss / len(dataLoader)
      losses.append(avg_loss)
      print(f"Epoch: {epoch+1} , Loss: {avg_loss:.4f}")


    # Create checkpoint directory if it doesn't exist
    os.makedirs("checkpoint", exist_ok=True)
    torch.save(model.state_dict() , f"checkpoint/model_epoch{epochs}.pth")
    
    # Save vocabularies to ensure consistency in predictions
    import json
    with open("checkpoint/vocab.json", "w", encoding="utf-8") as f:
        json.dump({
            "vocab_english": vocab_english,
            "vocab_hindi": vocab_hindi
        }, f, ensure_ascii=False, indent=4)
    print("Saved vocabulary to checkpoint/vocab.json")

    # Save loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), losses, marker='o', color='b', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/loss_plot.png')
    plt.close()
    print("Saved loss plot to loss_plot.png")
