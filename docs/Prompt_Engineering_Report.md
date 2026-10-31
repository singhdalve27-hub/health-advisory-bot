# Prompt Engineering & System Guardrails Report
**Project:** Health-Advisory Assistant
**Focus:** Mitigating hallucinations and enforcing clinical/safety constraints in medical-adjacent RAG applications.

## Overview
Because health advisories involve sensitive public safety information, preventing the AI from fabricating medical advice or treatments is critical. We implemented rigorous prompt architecture to govern model behavior.

## 1. Persona and Scope Definition
*   **Purpose:** Establishes the bot as an authoritative yet restricted public health assistant.
*   **System Instruction:** *"You are a helpful and accurate DOH Health-Advisory Assistant. Answer the user's question using ONLY the information provided in the context below."*
*   **Effect:** Prevents the model from pulling external, unverified medical knowledge from its pre-trained parameters.

## 2. The "Safe Fallback" Constraint
*   **Purpose:** Ensures the bot does not guess answers when context is missing, reducing the risk of harmful medical misinformation.
*   **System Instruction:** *"If the answer cannot be found in the context, respond exactly with: 'I don't have enough information to answer that based on the DOH manual.'"*
*   **Effect:** Creates a deterministic default failure state for out-of-scope medical inquiries.

## 3. Traceability and Citation Requirement
*   **Purpose:** Builds user trust by tying every response back to official documentation.
*   **System Instruction:** *"Always cite the specific disease or section if available in the context."*
*   **Effect:** Forces the model to explicitly reference the DOH manual source material.