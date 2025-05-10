# Agent.md

## Multi-Step LLM-Based Automatic Classifier for Policy Texts

### Motivation

This project aims to automate the annotation of new policy texts using a multi-step approach powered by large language models (LLMs). The goal is to support research into the evolution of EU climate policy, focusing on sectoral specificity and responsibility allocation.

### Approach

1. **Knowledge Injection**:  
   The LLM will be informed by the "Appendix and Codebook.pdf", especially the "III. Coding scheme" section, to understand the annotation categories and definitions.

2. **Input**:  
   The system will accept a single article (raw text) as input.

3. **Processing Steps**:  
   - **Step 1: Preprocessing**  
     The article is cleaned and segmented as needed.
   - **Step 2: LLM Annotation**  
     The LLM, prompted with the coding scheme, identifies relevant spans and assigns annotation categories (layer, feature, tag, start, stop, text).
   - **Step 3: Output Formatting**  
     The output is a JSON list of annotation objects, each with:
     - `layer`
     - `feature`
     - `tag`
     - `start`
     - `stop`
     - `text`  
     (No need to output tokens or span_id.)

4. **Example**:  
   Given the raw text of an article, the output will be a JSON array of annotation objects, similar to the provided Curated_Annotations.json, but without tokens or span_id.

### Notebook Integration

A notebook will be provided that:
- Accepts an article as input.
- Calls the LLM with the coding scheme context.
- Outputs the annotation JSON.

### Next Steps

- Extract the coding scheme from "Appendix and Codebook.pdf".
- Design LLM prompts for annotation.
- Implement the notebook pipeline.
