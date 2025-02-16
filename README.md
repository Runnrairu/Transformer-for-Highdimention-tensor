# Transformer-for-Highdimention-tensor
# 3次元テンソルを入力として受け付けるTransformerの実装
## 3次元テンソル版の魅力
現状通常のLLM系統のモデルは、過去の発言への注意を均等ないしあらかじめ決められた比率で行うことが多い。そこで三次元テンソルでの計算を考え、過去の発言すべてとそれに対する注目を計算していくことで、重要な指示をより長く保持し続けるシステムを作れるのではないかと考えた。

### 着想元
元はchatgptを使っていた時、何度私が「タメ口をやめろ」と言っても、時間が経つと慣れ慣れしくタメ口で反応してくるのにイラつくあまり、強く言った指示への注意をLLM側がずっと保持し続けるモデルを考えられないかと思ったのが最初である。

## アインシュタインの縮約
これを今回の実装における3次元テンソルの計算方法としたが、3次元テンソル型のtransformerの計算にはこれが最適などと現段階でいうつもりはない。もっとtransformer向きの3次元テンソル計算法が見つかる可能性は大いにある。

## アテンション考察
アテンション行列は「相関行列の拡張」と言える。そのためこれを有限集合×有限集合→実数の写像と考え、連続化すると2変数関数となるわけだが、これは「カーネル関数の拡張」になると考えている。
この思考は理論研究がメインで、直接の実装活用はかなり根本的なネットワークの改変を含むものだが、実装に応用しやすいアイデアとしても一つ用いる予定。

## 2/16更新
まだ三次元化はできていないが一応動くものを作れた。これを改良していきたいが一度作り直したほうがいい気がしてきた。
