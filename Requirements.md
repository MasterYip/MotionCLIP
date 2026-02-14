```bash
python -m src.visualize.text2motion ./exps/paper-model/checkpoint_0100.pth.tar --input_file assets/paper_texts.txt
```

```bash
proxychains pip install joblib smplx gdown chumpy-fork human_body_prior \
git+https://github.com/openai/CLIP.git \
# git+https://github.com/nghorbani/human_body_prior@cvpr19
```