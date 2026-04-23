# Time Series → Caption via Frozen VLM + Static Soft Prompt

## 0. Goal

Build a minimal and stable baseline for time-series caption generation using:

- Frozen Vision-Language Model (VLM)
- Static learnable soft prompt tokens
- Time series converted to image (line plot)

---

## 1. High-Level Pipeline

time series x
→ plot renderer (image)
→ frozen VLM encoder
→ [soft prompt tokens] + [text prompt] + [image embedding]
→ frozen LLM decoder
→ caption y

Trainable parameters:
- ONLY soft prompt tokens

Everything else:
- FROZEN

---

## 2. Data Format

### Input

Time series:
x ∈ ℝ^(T) or ℝ^(T × C)

### Output

Caption:
y = tokenized text

---

## 3. Time Series → Image Conversion

### Requirements

Convert each time series into a fixed-style line plot.

#### MUST FIX ALL VISUAL STYLE:

- background color
- axis range (or normalized)
- line width
- color
- resolution

### Example spec:

- image size: 100 × 100
- line color: black
- background: white
- no grid
- no legend
- x-axis evenly spaced
- Do Not visualize y-ticks

### Important

Do NOT introduce randomness in plotting
→ plotting must be deterministic

---

## 4. Model Architecture

### 4.1 Frozen VLM

Use VLM: Qwen/Qwen3-VL-8B-Instruct (You can see instruction of how to use this model on this website: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)


All weights:
requires_grad = False

---

### 4.2 Soft Prompt Tokens (Static)

Define:

P ∈ ℝ^(m × d)

Where:

- m = number of prompt tokens (e.g., 10–50)
- d = embedding dimension of LLM

Initialization:

P ~ Normal(0, 0.02)

Trainable:
requires_grad = True

---

### 4.3 Textual Instruction Prompt

Fixed text prompt:

"Generate a structured caption describing the time series."


---

## 5. Input Construction

Final input prompt to LLM:

[ P ; text prompt ]

---

## 6. Training Objective

Standard next-token prediction

---

## 7. Training Setup

### Training parameters

ONLY update:

P (soft prompt tokens)

Everything else:
frozen

---

## 8. Evaluation Protocol

- Generate caption and compared to the real caption in dataset

---