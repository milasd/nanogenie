# nanogenie

A reproduction of the world model Genie 1 architecture, as reported in the paper (Genie: Generative Interactive Environments)[https://arxiv.org/pdf/2402.15391], but in a much smaller scale.

I intend to reproduce the training process (in a "nano-scale"). For that purpose, I'm first building the following model components w/ PyTorch:
  I) Latent action model (infers the latent action 𝒂 between each pair of frames) 
  II) Video tokenizer (converts raw video frames into discrete tokens 𝒛)
  III) Dynamics model (given a latent action and past frame tokens, predicts the next frame of the video).

I would conceptually summarise the above in 2 "bigger blocks":
  1. The world model (Dynamics model, c): from input features z_t-1 and a, outputs z_t.
  2. Input feature "generation": From a set of frames, obtain frame features z_t-1 (with video tokenizer, II) and action a (with latent action inference, I).

After this step, I'll look into potential datasets of choice for the training process, which should be open-sourced. The to-be trained model for this study should have some million parameters, for a first experiment. 
