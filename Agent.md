# Agent.md

## Multi-Step LLM-Based Automatic Classifier for Policy Texts

### Motivation

This project aims to automate the annotation of new policy texts using a multi-step approach powered by large language models (LLMs). The goal is to support research into the evolution of EU climate policy, focusing on sectoral specificity and responsibility allocation.

### Approach

1. **Knowledge Injection**:  
   The LLM will be informed by the "Appendix and Codebook.pdf" via the coding scheme extracted to JSON, especially the "III. Coding scheme" section, to understand the annotation categories and definitions.

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

4. **Selective Annotation**:  
   To optimize computation and facilitate debugging, the system will support filtering by:
   - Specific layers (e.g., "Policydesigncharacteristics")
   - Specific tagsets (e.g., "Actor")
   - Combinations of layers and tagsets

5. **Evaluation Pipeline**:  
   The system will include an evaluation component that:
   - Compares LLM-generated annotations against manually curated annotations
   - Calculates precision, recall, and F1 scores for each layer/tagset
   - Reports agreement metrics for span identification and tag assignment
   - Generates confusion matrices for misclassifications

### Implementation Plan

1. **Core Annotation Functions**:
   - [x] Set up environment and dependencies
   - [ ] Implement coding scheme loading and filtering
   - [ ] Develop LLM prompting strategy with few-shot examples
   - [ ] Create span identification and classification logic
   - [ ] Build JSON output formatter

2. **Selective Layer/Tagset Processing**:
   - [ ] Create filter functions for layers and tagsets
   - [ ] Modify prompts to focus on specific annotation types
   - [ ] Add command-line/function parameters for filtering
   - [ ] Test with "Policydesigncharacteristics"/"Actor" only mode

3. **Evaluation Components**:
   - [ ] Implement annotation comparison logic
   - [ ] Create metrics calculation functions (precision, recall, F1)
   - [ ] Build visualization tools for results analysis
   - [ ] Develop report generation for evaluation results

4. **Code Organization**:
   - [ ] Create modular structure with separate Python files:
     - `utils.py`: Helper functions, path resolution, etc.
     - `annotation.py`: Core annotation functions
     - `evaluation.py`: Evaluation pipeline
     - `main.py`: CLI entry point
   - [ ] Maintain notebook for interactive exploration and demos

5. **Documentation**:
   - [ ] Add detailed docstrings to all functions
   - [ ] Create usage examples for different scenarios
   - [ ] Document evaluation metrics and interpretation

### Initial Testing Focus

For initial testing and development, focus on:
- Layer: "Policydesigncharacteristics"
- Tagset: "Actor"
- Sample article: "EU_32004L0008_Title_0_Chapter_0_Section_0_Article_01"

### Deliverables

1. A modular Python codebase for LLM-based policy text annotation
2. A demo notebook showing the annotation process
3. An evaluation report comparing LLM vs. human annotations
4. Documentation for usage and extension
