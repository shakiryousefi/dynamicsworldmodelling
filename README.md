# Readme
Code repository for project in the ETH Zurich – Deep Learning class 2023-2024. Much of the code is based on the DreamerV3 implementation in JAX. The notable differences are highlighted lower in the Readme file.

# Installation Instructions

The code in the repository was ran on the Euler cluster using Python 3.10. In order to install the needed packages, you can run
i

```sh
pip install -r requirements.txt
```

We also recommend upgrading JAX to utilize the GPU. This can be done by running


```sh
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

During our installation, we observed problems installing the Atari gym environment. Specifically, we got an error related to


```sh
error in gym setup command: 'extras_require' must be a dictionary whose values are strings or lists of strings containing valid project/version requirement specifiers.
```
We mitigated this by downgrading setuptools and wheel

```sh
pip install setuptools==65.5.0 "wheel<0.40.0"
```

# Running the code

We provide the bash scripts used to run the experiments for the project. The different hyperparameters are natively integrated into the DreamerV3 configuration -- allowing for being set in the configs.yaml file or directly overwritten using the terminal. An example input that overwrites the environment, alpha parameter, and k-steps can be seen below


```sh
python dreamerv3/train.py \
  --logdir ~/logdir/$(date "+%Y%m%d-%H%M%S") \
  --configs atari100k --task atari_pong --dynamics.k_steps 3 --dynamics.alpha_strength 5
```


# Code differences between DreamerV3

We highlight the differences between our implementation and DreamerV3 below. Specifically, we had two change the code in two main places: generating the mask, when training the dynamics predictor network, and calculating the actual loss.

The first part happens in dreamerv3/agent.py, where the differences between DreamerV3 and our implementation is:

```  
  def generate_mask(self, data, state = None):
    #Obtain the k and alpha from the configs list
    k = self.config.dynamics.k_steps
    alpha = self.config.dynamics.alpha_strength
    mask_list = []
    
    #Loop through the batch
    for batch_index in range(data.shape[0]):
        batch = data[batch_index]
        n = batch.shape[0]
        for i in range(n):
            temp_mask = jnp.zeros(batch.shape[1:], dtype=int)
            #Adjust for the fact that we need to truncate
            for j in range(max(0, i - k), min(n, i + k + 1)):
                temp_mask += (jnp.abs(batch[i] - batch[j]) > 0).astype(int)
            temp_mask = (temp_mask > 0).astype(int)
            mask_list.append(temp_mask)
    
    mask = jnp.array(mask_list).reshape(data.shape)

    #Map all the moving pixels to alpha, and the rest to ones
    adjusted_mask = jnp.where(mask > 0, alpha, 1)

    return adjusted_mask

  def loss(self, data, state):
    embed = self.encoder(data)
    prev_latent, prev_action = state
    prev_actions = jnp.concatenate([
        prev_action[:, None], data['action'][:, :-1]], 1)
    post, prior = self.rssm.observe(
        embed, prev_actions, data['is_first'], prev_latent)
    dists = {}
    feats = {**post, 'embed': embed}
    for name, head in self.heads.items():
      out = head(feats if name in self.config.grad_heads else sg(feats))
      out = out if isinstance(out, dict) else {name: out}
      dists.update(out)
    losses = {}
    losses['dyn'] = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss)
    losses['rep'] = self.rssm.rep_loss(post, prior, **self.config.rep_loss)
    for key, dist in dists.items():
      loss = 0
      #Check if the output of the predictor is the frame 
      if key == 'image':
          mask = None
          #Check if we should use masked loss, and generate the mask using the auxillary function above
          if self.config.dynamics.masked_loss:
              mask = self.generate_mask(data[key])
          loss = -dist.log_prob(data[key].astype(jnp.float32), mask)
```

Secondly, as we can see above the mask is passed to the loss function, which requirees that we edit the loss. This is done in dreamerv3/jaxutils.py, where


```
  def log_prob(self, value, mask = None):
    assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
    distance = ((self._mode - value) ** 2)

    # Apply the window mask if provided
    if mask is not None:
        # Ensure that the mask is broadcastable to the shape of 'distance'
        distance *= mask  # Element-wise multiplication

    if self._agg == 'mean':
      loss = distance.mean(self._dims)
    elif self._agg == 'sum':
      loss = distance.sum(self._dims)
    else:
      raise NotImplementedError(self._agg)
    return -loss
```
