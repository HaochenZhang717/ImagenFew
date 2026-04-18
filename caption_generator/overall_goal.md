# Latent-Conditioned Caption Generation for Time Series

## Overview

We propose a two-stage generative framework to model and generate structured captions for time series data. The pipeline first learns a latent representation of time series using a Variational Autoencoder (VAE), and then conditions a Large Language Model (LLM) on this latent representation to generate captions. A diffusion model is further trained over the latent space to enable unconditional generation of captions.

In addition to latent conditioning, we also incorporate **natural language prompts** as auxiliary inputs to guide the generation process.

---

## Motivation

The goal is to generate captions that follow the distribution of a given time series dataset. Instead of directly modeling the caption distribution, we introduce a latent variable that captures the underlying structure of time series and use it as a bridge between time series and language.

This enables:

- Learning a shared latent space for time series and captions
- Generating captions without explicitly conditioning on a specific time series
- Capturing dataset-level semantics through latent diffusion
- Leveraging natural language prompts to guide or constrain generation

---

## Pipeline Overview

The framework consists of two main stages:

1. **Latent Representation Learning + Caption Alignment**
2. **Latent Distribution Modeling via Diffusion**

---

## Stage 1: Two-Phase VAE + Latent-Conditioned Caption Generation

### 1.1 Phase A: VAE Reconstruction Pretraining

We first train a Variational Autoencoder (VAE) on time series data using only reconstruction and KL regularization:

- Encoder:
  
  \[
  z \sim q_\phi(z | x)
  \]

- Decoder:
  
  \[
  \hat{x} \sim p_\theta(x | z)
  \]

- Objective:

  \[
  \mathcal{L}_{VAE} = \mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x|z)] - \beta \cdot KL(q_\phi(z|x) \| p(z))
  \]

where:
- \( x \): time series
- \( z \): latent variable
- \( p(z) = \mathcal{N}(0, I) \)

In this phase, the goal is purely to learn a meaningful latent space for time series structure. No language modeling objective is used yet.

---

### 1.2 Phase B: Latent as Soft Prompt

After the VAE has learned an initial latent space, we use the latent variable \( z \) as a **soft prompt** for a Large Language Model (LLM).

A projection module maps \( z \) into a sequence of embeddings:

\[
h = P(z) \in \mathbb{R}^{m \times d}
\]

where:
- \( m \): number of soft prompt tokens
- \( d \): LLM embedding dimension

These embeddings are prepended to the token embeddings of the textual input.

---

### 1.3 Additional Natural Language Prompt

In addition to the latent soft prompt, we introduce a **textual prompt** \( c \) to guide generation:

\[
c = \{ \text{instruction or context prompt} \}
\]

Examples of such prompts include:

- "Generate a structured caption for the following time series."
- "Describe the trend, distribution, and periodicity."

The final input to the LLM is composed of:

\[
[\; h \; ; \; \text{Embed}(c) \; ; \; \text{Embed}(y_{<t}) \;]
\]

This design enables:

- Combining **latent semantics** (from time series) with **explicit language guidance**
- Controlling generation style and structure
- Improving stability and consistency of captions

---

### 1.4 Phase B: Joint VAE + LLM Training

We then jointly train the VAE encoder-side latent representation together with the LLM caption model, conditioned on both latent soft prompts and textual prompts:

\[
p_\psi(y | z, c)
\]

Caption objective:

\[
\mathcal{L}_{cap} = - \mathbb{E}_{(x,y)} \left[ \log p_\psi(y | z, c) \right]
\quad \text{where } z \sim q_\phi(z|x)
\]

In the second phase, we do **not** keep optimizing the reconstruction term. Instead, we optimize caption prediction together with an optional KL regularization term:

\[
\mathcal{L}_{joint} = \lambda_{cap} \cdot \mathcal{L}_{cap} + \lambda_{KL} \cdot KL(q_\phi(z|x)\|p(z))
\]

Training details:

- Input: soft prompt \( h \) + prompt text \( c \) + caption tokens
- Target: caption sequence
- Optimization: next-token prediction loss (teacher forcing) with optional KL regularization
- Reconstruction loss is not included in this phase

---

### 1.5 Overall Stage 1 Procedure

Stage 1 is therefore split into two sequential phases:

1. **Phase A**: train the VAE with reconstruction + KL
2. **Phase B**: jointly train VAE latent alignment and caption generation with caption loss + optional KL

This ensures that the latent variable \( z \):

- First encodes time series structure through reconstruction learning
- Then becomes aligned with caption generation
- Works jointly with textual prompts for controlled generation

---

## Stage 2: Diffusion Model on Latent Space

### 2.1 Goal

Learn a generative model over latent variables:

\[
z \sim p(z)
\]

so that we can sample new latent variables without conditioning on input time series.

---

### 2.2 Diffusion Training

We train a diffusion model on latent samples \( z \sim q_\phi(z|x) \).

Forward process:

\[
q(z_t | z_0)
\]

Reverse model:

\[
p_\theta(z_{t-1} | z_t)
\]

Training objective:

\[
\mathcal{L}_{diff} = \mathbb{E}_{z_0, t, \epsilon} \left[ \| \epsilon - \epsilon_\theta(z_t, t) \|^2 \right]
\]

where:
- \( z_0 \): latent from VAE encoder
- \( z_t \): noisy latent
- \( \epsilon \): Gaussian noise

---

## Inference

### Case 1: Caption Generation from Time Series

Given a time series \( x \):

1. Encode:

   \[
   z \sim q_\phi(z|x)
   \]

2. Project to soft prompt:

   \[
   h = P(z)
   \]

3. Provide textual prompt \( c \)

4. Generate caption:

   \[
   y \sim p_\psi(y | z, c)
   \]

---

### Case 2: Unconditional Caption Generation

1. Sample latent from diffusion:

   \[
   z \sim p(z)
   \]

2. Project to soft prompt:

   \[
   h = P(z)
   \]

3. Provide textual prompt \( c \)

4. Generate caption:

   \[
   y \sim p_\psi(y | z, c)
   \]

---

## Key Design Choices

- **Latent variable as interface** between time series and language
- **Soft prompt conditioning** via projected latent embeddings
- **Natural language prompt conditioning** for controllability
- **Diffusion prior** to model dataset-level latent distribution
- **Two-phase Stage 1 training**: reconstruction pretraining followed by joint caption alignment

---

## Potential Extensions

- Use sequence latent instead of a single vector
- Replace VAE with hierarchical latent models
- Add conditional diffusion for controlled generation
- Learn prompt tuning jointly with latent projection
- Explore different prompt templates for better alignment

---

## Summary

This framework models caption generation through a latent variable learned from time series. By combining VAE-based representation learning, LLM-based conditional generation (via both soft prompts and textual prompts), and diffusion-based latent modeling, we enable both conditional and unconditional caption generation aligned with the dataset distribution.
