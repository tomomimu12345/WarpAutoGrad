# Warp-PyTorch Integration Sample

## 概要
このサンプルコードは、**NVIDIA Warp** と **PyTorch** を組み合わせて、自作カーネルの勾配計算を行い、パラメータを更新する例です。  
Warp の `Tape` 機能を利用して、GPU 上で実行される Warp カーネルの結果を用いて、PyTorch の `torch.nn.Parameter` を更新します。
---

## 必要環境
- Python 3.9+
- CUDA 対応 GPU
- PyTorch
- NVIDIA Warp

```bash
pip install torch warp-lang
```

## 参考
- https://github.com/NVIDIA/warp
