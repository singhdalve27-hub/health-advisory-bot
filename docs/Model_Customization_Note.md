# Model Customization Note: Parameter-Efficient Fine-Tuning (PEFT/LoRA/QLoRA) Analysis vs. RAG
**Course:** IS Professional Elective 4 — Generative AI Systems  
**Checkpoint 4 Deliverable 15 (Final)**  
**Project:** DOH Health-Advisory Assistant  
**Authors:** Dalve Singh, Gian Garcia, Lonnette Oroyan  

---

## 1. Introduction & Problem Context
When deploying generative AI models into clinical and public health advisory applications, engineers encounter a fundamental architectural choice:
1. **Model Adaptation via Fine-Tuning:** Modifying model weights through full fine-tuning or Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA and QLoRA.
2. **Context-Grounded Retrieval (RAG):** Supplying factual domain knowledge dynamically at inference time without modifying underlying model weights.

This note provides a technical assessment of PEFT for the DOH Health-Advisory Assistant, evaluating theoretical and empirical considerations regarding LoRA/QLoRA, and articulating a well-reasoned justification for using an instruction-tuned foundation model (Gemini 3.8 Flash) as-is within a RAG architecture.

---

## 2. Technical Evaluation of PEFT (LoRA / QLoRA)

### 2.1 Theoretical Foundations of PEFT
Parameter-Efficient Fine-Tuning freezes the pre-trained model weights $W_0 \in \mathbb{R}^{d \times k}$ and introduces low-rank decomposition matrices $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times k}$ where the rank $r \ll \min(d, k)$:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$

* **LoRA (Low-Rank Adaptation):** Trains only the adapter parameters $A$ and $B$, typically reducing trainable parameters by $99\%$ while maintaining task performance.
* **QLoRA (Quantized Low-Rank Adaptation):** Extends LoRA by quantizing the base model weights to 4-bit NormalFloat (NF4) with Double Quantization and paged optimizers, allowing fine-tuning of 7B-13B parameter models on a single consumer GPU (e.g., 16GB VRAM).

### 2.2 Potential Benefits of PEFT for This Application
If fine-tuning were applied to our health advisory assistant, the primary advantages would be:
1. **Domain-Specific Tone and Formatting:** Conditioning the model to output clinical triage notes strictly following Philippine DOH terminology (e.g., Tagalog colloquialisms like *tigdas*, *pulmonya*, *aso/pusa kagat*, *tandok*).
2. **Deterministic Output Structure:** Enforcing standardized JSON schemas or triage bullet templates directly through the model's learned output distribution rather than prompt steering.
3. **Reduced Prompt Token Overhead:** Baking recurring system instructions into model weights rather than transmitting 500+ tokens of system instructions on every API turn.

---

## 3. Why Base/Instruction-Tuned Model (As-Is) + RAG was Chosen

Despite the theoretical advantages of PEFT, we adopted the **base instruction-tuned model (Gemini 3.8 Flash) as-is**, paired with RAG. Our technical justification is based on five critical factors:

### 3.1 The "Parametric vs. Non-Parametric" Knowledge Dilemma (Hallucination Risk)
* In healthcare, safety is paramount. When factual knowledge is baked into model weights (parametric memory), neural networks generate responses probabilistically. Even a fine-tuned medical model can hallucinate subtle but lethal inaccuracies (e.g., misstating an incubation period or pediatric medication dosage).
* In contrast, RAG provides **non-parametric verifiable grounding**. The model is constrained to extract information only from retrieved context chunks. If information is missing, our system deterministically triggers the exact fallback refusal string:
  > *"I don't have enough information to answer that based on the DOH manual."*

### 3.2 Dynamic Knowledge Freshness and Maintainability
* Public health advisories evolve constantly during disease outbreaks (e.g., updated quarantine guidelines, novel dengue serotype warnings, new vaccine schedules).
* With fine-tuning, every policy update requires dataset re-annotation, model retraining, validation, and redeployment—a costly and slow process.
* With RAG and our automated ingestion pipeline (`src/ingestion.py`), updating the knowledge base takes **seconds**: the health administrator simply uploads the new PDF advisory, which is instantly parsed, chunked, embedded, and indexed into ChromaDB.

### 3.3 Catastrophic Forgetting & General Reasoning Degradation
* Fine-tuning smaller models (e.g., Llama-3-8B or Mistral-7B) on specialized medical texts frequently causes **catastrophic forgetting**, degrading the model's instruction-following ability, multi-turn reasoning, and safety alignment.
* High-capability foundation models like Gemini 3.8 Flash possess state-of-the-art conversational reasoning, context synthesis, and multilingual translation capabilities out-of-the-box, which are preserved without degradation.

### 3.4 Operational Simplicity & Infrastructure Costs on Railway
* Deploying a fine-tuned 7B model in production requires a GPU-enabled cloud instance (such as an AWS EC2 `g4dn.xlarge` or similar) costing $\$300+$ monthly.
* Our RAG architecture decouples retrieval (lightweight CPU-based all-MiniLM-L6-v2 embeddings running locally inside a 1.2GB Docker container on Railway) from generation (Gemini API serverless calls). This reduces infrastructure operational costs to near zero while maintaining sub-2-second latency.

---

## 4. Proposed Hybrid Roadmap (If PEFT Were Implemented)
Should the project scale beyond the capstone phase, the optimal architecture would be a **hybrid RAG + QLoRA model**:
1. **Target of QLoRA:** Target exclusively query reformulation, medical terminology alignment (English-Filipino/Tagalog code-switching), and refusal classification.
2. **Dataset:** 1,500 curated patient-assistant dialogue pairs with Philippine community health terms.
3. **Execution:** Freeze all attention layers except $W_q, W_v$ projections with rank $r=16$, $\alpha=32$.
4. **Retention of RAG:** The fine-tuned model would *still* query ChromaDB for grounding context, ensuring zero ungrounded clinical claims.

---

## 5. Conclusion
For the scope, safety requirements, and deployment goals of the DOH Health-Advisory Assistant on Railway, utilizing the instruction-tuned Gemini 3.8 Flash model as-is within a strictly guarded LangChain RAG pipeline provides superior factual accuracy, auditability, update agility, and cost-efficiency compared to isolated parameter fine-tuning.

