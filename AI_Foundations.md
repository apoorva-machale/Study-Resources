# 📘 OCI AI Foundations – Beginner-Friendly Notes

> These notes summarize my understanding and key learnings from the **OCI AI Foundations** course.

---

## 🤖 What is AI?

Artificial Intelligence (AI) is the ability of machines to imitate human intelligence, including cognitive abilities and problem-solving skills.

- **Artificial General Intelligence (AGI):** Can replicate human intelligence capabilities, such as learning new skills, thinking abstractly, and communicating. Applying AGI to solve problems in machines is what we generally refer to as AI.

**Why do we need AI?**
- Automation and decision-making
- Creative support

**AI Domains and Examples:**
- Language: Translation
- Vision: Image classification
- Speech: Text-to-speech
- Product recommendations: Cross-selling
- Anomaly detection: Fraudulent transaction detection
- Reinforcement learning: Self-driving cars
- Forecasting: Weather predictions
- Generating content: Images from text, etc.

---

## 🌐 AI vs ML vs DL

The relationship is nested: **DL ⊂ ML ⊂ AI**. See the diagram below.

| AI vs ML vs DL (Nested) | ML Types |
|---|---|
| ![AI vs ML vs DL](images/ai_ml_dl_nested.png) | ![ML Types](images/ml_types_corrected.png) |

**Definitions:**
- **AI:** Broad field of making machines intelligent.
- **ML:** Subset of AI where systems learn from data.
- **DL:** Subset of ML using neural networks with multiple layers.

**ML Categories:**
- **Supervised Learning:** Labeled data. Regression (continuous output), Classification (categorical output)
- **Unsupervised Learning:** Unlabeled data. Clustering (group similar items), Dimensionality Reduction (find structure)
- **Reinforcement Learning:** Trial-and-error learning with agent–environment–reward setup

**Unsupervised Workflow:**

![Unsupervised Workflow](images/unsupervised_workflow.png)

---

## 🧠 Neural Networks (ANN)

![ANN Architecture](images/ann_architecture_deep.png)

**Components:** Input, hidden, output layers; neurons; weights; bias; activation functions

**Training (Backpropagation):**
1. Make a prediction
2. Compare with target → compute error
3. Adjust weights (gradient descent)
4. Repeat over many examples

**Advantages:** Feature extraction, parallel processing, scalability, better performance

**Brief History:**
- 1950s: Artificial neurons, perceptron, MLP
- 1990s: CNNs
- 2010: GPUs speed training
- 2012: AlexNet, DQN
- 2016: GANs, Transformers
- Today: LLMs, Diffusion Models

---

## 🧩 RNNs, LSTMs, CNNs

| CNN Workflow | LSTM Workflow |
|---|---|
| ![CNN Workflow](images/cnn_workflow.png) | ![LSTM Workflow](images/lstm_workflow.png) |

**RNN Sequence Mapping Types:**
![RNN Mapping Types](images/rnn_mapping_types.png)
- One-to-One: Single input → single output (e.g., image → label)
- One-to-Many: Single input → sequence (e.g., image → caption)
- Many-to-One: Sequence → single output (e.g., text → sentiment)
- Many-to-Many: Sequence → sequence (e.g., translation, NER)

**CNN:** Input → Convolution + activation → Pooling → Fully connected → Output
- Applications: Image classification, segmentation, face recognition, medical imaging, autonomous vehicles
- Limitations: Overfitting, sensitivity, high computation, interpretability issues

**LSTM:** Selectively remembers or forgets information via gates; handles long-range dependencies and mitigates vanishing gradients

---

## 🎨 Generative AI

- Creates new data following training data patterns
- Difference from ML: ML predicts known outputs; GenAI generates new content
- Types: Text-based, multimodal (text, images, audio, video)
- Applications: Content generation, medical imaging, NLP, creative tasks

---

## 📚 LLMs and Transformers

**LLM:** Probabilistic model of text sequences; predicts likely next words
- Example: “I am going to a birthday party. I took a ___ with me.” → likely “gift”

**Transformers:** Process tokens in parallel, model long-range dependencies via self-attention
- Encoder builds contextual representations
- Decoder generates outputs with cross-attention

![Transformer Overview](images/transformer_overview.png)

---

## ✍️ Prompt Engineering

- Refine input prompts to elicit desired outputs
- Types: In-context, k-shot, chain-of-thought
- Mitigation of hallucinations: RAG, verification, constraints

---

## ⚙️ Customizing LLMs

- **Prompt Engineering:** Optimize inputs
- **RAG:** Augment model with external knowledge without modifying model
- **Fine-Tuning:** Adapt pre-trained models with custom datasets
- **Inference:** Generate outputs using trained models

**Benefits:** Better performance, domain adaptation, efficiency


