# nanogenie

A reproduction of the world model Genie 1 architecture, as reported in the paper (Genie: Generative Interactive Environments)[https://arxiv.org/pdf/2402.15391], but in a much smaller scale.

I intend to reproduce the training process (in a "nano-scale"). For that purpose, I'm first building the following model components w/ PyTorch:

  - I) Latent action model (infers the latent action 𝒂 between each pair of frames) 
  - II) Video tokenizer (converts raw video frames into discrete tokens 𝒛)
  - III) Dynamics model (given a latent action and past frame tokens, predicts the next frame of the video).


### Training

I would conceptually summarise the above in 2 "bigger blocks" for personal understanding on how to map my development process:
  1. Input feature "generation": From a set of frames, obtain frame features z_t-1 (with video tokenizer, II) and action a (with latent action inference, I).
  2. The world model (Dynamics model, III): from input features z_t-1 and a, outputs z_t.

+------------------+     +-------------+
| Input features   | --> | World model |
| "generation"     |     |    (III)    |
|   (I, II)        |     |             |
+------------------+     +-------------+


After this step, I'll look into potential datasets of choice for the training process, which should be open-sourced. The to-be trained model for this study should have some million parameters, for a first experiment. 

### Inference

After the training step is completed, the inference part shall be developed. I summarise the concept as encoder --> Dynamics model (world model, III) --> decoder. Note that the world model output features z_t are iteratively used as input for subsequent step z_t+1... etc
