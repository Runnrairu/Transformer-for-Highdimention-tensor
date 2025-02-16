import torch
import torch.nn as nn
import torch.optim as optim

# ボキャブラリを定義（サンプル用）
en_vocab = {'hello': 0, 'world': 1, '<pad>': 2, '<sos>': 3, '<eos>': 4}
fr_vocab = {'bonjour': 0, 'monde': 1, '<pad>': 2, '<sos>': 3, '<eos>': 4}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# テンソル変換関数
def text_to_tensor(text, vocab):
    return torch.tensor([vocab.get(word, vocab['<pad>']) for word in text.split()], dtype=torch.long)

# サンプルのデータセット
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, en_sentences, fr_sentences):
        self.en_sentences = en_sentences
        self.fr_sentences = fr_sentences
    
    def __len__(self):
        return len(self.en_sentences)
    
    def __getitem__(self, idx):
        return self.en_sentences[idx], self.fr_sentences[idx]

# モデルの定義
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.transformer = nn.Transformer(hidden_dim, num_layers)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, src, tgt):
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        output = self.transformer(src_emb, tgt_emb)
        output = self.fc_out(output)
        return output

# 訓練関数
def train(model, dataset, num_epochs=10, batch_size=32):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=en_vocab['<pad>'])
    optimizer = optim.Adam(model.parameters())
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        for i, (src, tgt) in enumerate(data_loader):
            # テキストをテンソルに変換
            src_tensor = [text_to_tensor(text, en_vocab) for text in src]
            tgt_tensor = [text_to_tensor(text, fr_vocab) for text in tgt]

            # 最大の長さを取得
            max_len_src = max(len(x) for x in src_tensor)
            max_len_tgt = max(len(x) for x in tgt_tensor)

            # パディングを適用
            src_tensor = [torch.cat([x, torch.tensor([en_vocab['<pad>']] * (max_len_src - len(x)))]).to(device) for x in src_tensor]
            tgt_tensor = [torch.cat([x, torch.tensor([fr_vocab['<pad>']] * (max_len_tgt - len(x)))]).to(device) for x in tgt_tensor]

            # パディング後のテンソルをスタック
            src_tensor = torch.stack(src_tensor).to(torch.long)  # <-- ここで LongTensor に変換
            tgt_tensor = torch.stack(tgt_tensor).to(torch.long)  # <-- ここで LongTensor に変換

            optimizer.zero_grad()

            # モデルの出力と損失計算
            output = model(src_tensor, tgt_tensor)
            loss = criterion(output.view(-1, len(fr_vocab)), tgt_tensor.view(-1))

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# データセットの準備（サンプル）
en_sentences = ['hello world', 'hello', 'world']
fr_sentences = ['bonjour monde', 'bonjour', 'monde']
dataset = SimpleDataset(en_sentences, fr_sentences)

# モデルの定義
input_dim = len(en_vocab)
output_dim = len(fr_vocab)
hidden_dim = 16
num_layers = 2
model = TransformerModel(input_dim, output_dim, hidden_dim, num_layers).to(device)

# 訓練開始
train(model, dataset)
