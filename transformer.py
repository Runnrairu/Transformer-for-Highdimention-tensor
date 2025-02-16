import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from datasets import load_dataset

# GPUが使える場合はGPU、そうでなければCPUを使用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データセットのロード
dataset = load_dataset('wmt14', 'en-fr')

# トークン化器の設定
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

# 語彙の構築
def build_vocab(data):
    return build_vocab_from_iterator(map(tokenizer, data), specials=["<unk>", "<pad>", "<bos>", "<eos>"])

# 英語とフランス語の語彙作成
en_vocab = build_vocab(dataset['train']['translation']['en'])
fr_vocab = build_vocab(dataset['train']['translation']['fr'])

# テキストをインデックスに変換
def text_to_tensor(text, vocab):
    return torch.tensor([vocab[token] for token in tokenizer(text)], dtype=torch.long)

# Transformerモデルの定義
class TransformerModel(nn.Module):
    def __init__(self, en_vocab_size, fr_vocab_size, emb_size=512, nhead=8, num_layers=6):
        super(TransformerModel, self).__init__()
        
        # エンコーダとデコーダの入力埋め込み層
        self.embedding_en = nn.Embedding(en_vocab_size, emb_size)
        self.embedding_fr = nn.Embedding(fr_vocab_size, emb_size)
        
        # Transformer モジュール
        self.transformer = nn.Transformer(d_model=emb_size, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        
        # 出力層（翻訳結果を生成）
        self.fc_out = nn.Linear(emb_size, fr_vocab_size)
    
    def forward(self, src, tgt):
        # 埋め込み層を通してトークンを変換
        src_emb = self.embedding_en(src)
        tgt_emb = self.embedding_fr(tgt)
        
        # Transformerを通す
        output = self.transformer(src_emb, tgt_emb)
        
        # 出力層を通して結果を得る
        output = self.fc_out(output)
        
        return output

# モデル作成
model = TransformerModel(len(en_vocab), len(fr_vocab)).to(device)

# 損失関数と最適化手法
criterion = nn.CrossEntropyLoss(ignore_index=en_vocab['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 学習ループ
def train(model, dataset, num_epochs=10, batch_size=32):
    model.train()
    for epoch in range(num_epochs):
        for i in range(0, len(dataset['train']), batch_size):
            # バッチのサンプルを取得
            src = dataset['train']['translation']['en'][i:i+batch_size]
            tgt = dataset['train']['translation']['fr'][i:i+batch_size]
            
            # テンソルに変換
            src_tensor = torch.stack([text_to_tensor(text, en_vocab) for text in src]).to(device)
            tgt_tensor = torch.stack([text_to_tensor(text, fr_vocab) for text in tgt]).to(device)
            
            # パディングを含むバッチを0でパディング
            src_tensor = torch.nn.functional.pad(src_tensor, (0, 1))
            tgt_tensor = torch.nn.functional.pad(tgt_tensor, (0, 1))
            
            optimizer.zero_grad()
            
            # 順伝播
            output = model(src_tensor, tgt_tensor)
            
            # 損失計算
            loss = criterion(output.view(-1, len(fr_vocab)), tgt_tensor.view(-1))
            
            # バックプロパゲーション
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# 学習開始
train(model, dataset)

# 翻訳関数
def translate(model, text, en_vocab, fr_vocab):
    model.eval()
    # テキストをテンソルに変換
    src_tensor = text_to_tensor(text, en_vocab).unsqueeze(0).to(device)
    
    # 出力を生成
    with torch.no_grad():
        output = model(src_tensor, src_tensor)  # 仮に同じテンソルを入力として使用
    
    # 出力をフランス語の単語に変換
    translated_text = [fr_vocab.lookup_itos[idx] for idx in output.argmax(dim=-1).squeeze().tolist()]
    return " ".join(translated_text)

# 翻訳
sample_text = "This is a test sentence."
translated = translate(model, sample_text, en_vocab, fr_vocab)
print(translated)