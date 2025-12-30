
### Motion2Text

I hope you implement #file:motion2text.py  file and update #file:visualize.py  to add a demo for motion2text.

Requirements:
1. retrieve data of motion according to #file:paper_motion2text.csv  (and implement it with examples)
2. Encode motion to latent features
3. Decode using clip to text, and print in terminal.
4. update #file:README.md  for the new example

## Args Loading Refactor
I found the arg parse method used in this repo are too messy. Args are separated in #file:opt.yaml , default value in src/parser. Now I hope to create a clean model init method, requirement:
1. Add a new load func get_motion_clip in #file:get_model_and_data.py , which use hydra to load hierarchical args from #file:default_cfg.yaml (do not use argparse) and init model with encoder, decoder and clip_model.
2. Implement #file:minimal_load.py  as an minimal load example.
3. Because this only loads the model, do not include any other (dataset / visualization) args