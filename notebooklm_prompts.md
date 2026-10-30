# Google NotebookLM Output Planning & Prompt Engineering Guide
## Interactive Feature Recipes & Prompt Blueprints for Source-Grounded Learning
**Organization:** Bharat Space Education Research Centre (BSERC) – GenAI Internship  
**Project:** Data Science & AI Knowledge Notebook – A Source-Grounded Learning Assistant  
**Platform:** Google NotebookLM (`notebook.google.com`)  

---

## 1. Overview & Setup Strategy

Google NotebookLM is a source-grounded generative AI notebook developed by Google. Unlike general-purpose chatbots that generate answers from ungrounded memory weights, NotebookLM anchors its generation strictly to the sources you upload, appending clickable inline citations (`[1]`, `[2]`) that trace every claim directly back to the original text.

This guide provides tested, high-precision prompts designed for the **10 target educational outputs** specified by BSERC, alongside workflows for testing every feature in the NotebookLM interface.

### Recommended NotebookLM Chat Configuration
Before issuing prompts in the chat box, configure the notebook's persona and objective:
1. In the NotebookLM bottom chat bar, click **Configure Chat** (or notebook settings icon).
2. **Chat Persona / Role:** Set to *"Expert Data Science Educator, Research Mentor, and Academic Tutor for BSERC"*.
3. **Instruction Guidelines:**  
   *"You are a strictly source-grounded educational assistant. Answer questions strictly using the uploaded sources. Every factual statement, equation, or code implementation must be substantiated by an inline citation to the source. If a requested concept is not covered in the active sources, explicitly state: 'This information is not supported by the uploaded sources' rather than hallucinating."*

---

## 2. Prompts for the 10 Required NotebookLM Outputs

---

### Output 1: Comprehensive Study Guide
- **NotebookLM Location:** Studio Panel -> Create -> Study Guide (or paste prompt in Chat and save to note).
- **Purpose:** In-depth, curriculum-aligned academic reference manual for graduate and undergraduate students.
- **Prompt:**
```markdown
Act as an expert Data Science professor at BSERC. Synthesize a Comprehensive Study Guide across all uploaded sources covering:
1. Fundamentals of Data Science and the Empirical Lifecycle
2. Python for Data Science and Memory Execution Models
3. NumPy Vectorization and Pandas Tabular Wrangling
4. Data Cleaning, Imputation Mechanisms, and Preprocessing Pipelines
5. Data Visualization Theory and Graphical Encodings
6. Statistics, Hypothesis Testing, and Exploratory Data Analysis
7. Supervised, Unsupervised, and Reinforcement Learning Fundamentals
8. Generative AI, Transformers, and Large Language Models
9. Mission-Critical Applications in Space Science and Industry
10. Data Science Career Roles, Pathways, and Engineering Rigor

For every section, provide:
- A rigorous theoretical explanation grounded in the text.
- Key mathematical formulations with variable definitions.
- Practical Python/NumPy/Pandas/Scikit-learn code implementations where applicable.
- Common student pitfalls and misconceptions.
- Five self-check revision questions.

Ensure every major assertion includes an exact inline source citation.
```

---

### Output 2: Revision Notes
- **NotebookLM Location:** Studio Panel -> Create -> Study Guide / Notes (or Chat -> Save as Note).
- **Purpose:** High-density, rapid-review cheat sheet for exam and interview preparation.
- **Prompt:**
```markdown
Generate a high-yield, high-density Revision Notes document summarizing the core tenets of Data Science and Machine Learning from the uploaded sources.

Structure the notes into:
1. Core Definitions (one-sentence definitions for the 25 most critical terms).
2. Essential Mathematical Formulas (Loss functions, Gradient Descent, Normal equation, Central Limit Theorem, Attention mechanism, and Bessel's correction).
3. The Bias-Variance Trade-Off and Regularization Matrix (Comparing L1 vs. L2).
4. The Golden Rules of Preprocessing (Preventing data leakage, choosing proper scalers, and handling missingness mechanisms).
5. Quick Diagnostic Cheat Sheet (Symptoms, root causes, and fixes for Overfitting, Underfitting, Vanishing Gradients, and Covariate Shift).

Keep the format bulleted, concise, and heavily cited. Avoid filler prose.
```

---

### Output 3: Flashcards
- **NotebookLM Location:** Studio Panel -> Flashcards (or Chat generation formatted for cards).
- **Purpose:** Active-recall testing cards for rapid spaced-repetition study.
- **Prompt:**
```markdown
Create a set of 20 high-yield, two-sided Flashcards based strictly on the uploaded sources to test an MS Data Science student.

Format each flashcard using this exact markdown schema:
### Flashcard [Number]: [Topic Area]
- **Front (Question / Prompt):** [Clear, challenging conceptual or mathematical question]
- **Back (Answer / Rationale):** [Direct, rigorous answer citing specific source text [X], followed by the core formula or mechanism]

Cover a balanced mix:
- 4 cards on Python, NumPy memory layout, and Pandas indexing.
- 4 cards on Data Cleaning, Leakage, and Outlier Fences.
- 4 cards on Statistical Inference, CLT, and Hypothesis Testing.
- 4 cards on Classical ML, Bias-Variance, and Classification Metrics.
- 4 cards on Transformers, Self-Attention, and Generative AI RAG.
```

---

### Output 4: Interactive Quiz
- **NotebookLM Location:** Studio Panel -> Quiz.
- **Purpose:** Diagnostic self-assessment with citation-backed feedback.
- **Prompt:**
```markdown
Generate a challenging 10-question multiple-choice diagnostic Quiz based on the uploaded data science materials.

Requirements for each question:
1. Provide a clear, realistic scenario or conceptual problem.
2. Provide four distinct answer options (A, B, C, D).
3. Below each question, provide a collapsed or separated Answer Section including:
   - Correct Answer.
   - Comprehensive Grounded Explanation citing the exact source and page/section.
   - Distractor Analysis: Explicitly explain why the other three choices are incorrect or represent common misconceptions.
```

---

### Output 5: Audio Overview (Deep Dive Podcast)
- **NotebookLM Location:** Studio Panel -> Audio Overview -> Generate (Customize).
- **Purpose:** An engaging, two-host conversational "deep dive" podcast discussing data science concepts.
- **Customization Prompt:**
```text
Host Directions:
Produce a conversational, intellectually stimulating deep dive discussion between two expert AI researchers. 
Focus the conversation on: "Why Source Grounding and Rigorous Statistics Matter in the Age of Generative AI."

Key topics to debate and explain:
1. The transition from classical data science and exploratory statistics to massive Generative AI foundation models.
2. Why LLMs hallucinate, and how source-grounded architectures like NotebookLM and RAG solve this.
3. How real-world aerospace and space research at BSERC relies on telemetry anomaly detection and strict validation rather than ungrounded intuition.
4. Advice for students transitioning into Machine Learning Engineering.

Tone: Enthusiastic, academic yet accessible, intellectually rigorous.
```
*Interactive Join-In Feature:* Once the audio is generated, click the **"Join In"** (or interactive conversation) button to ask the audio hosts real-time clarifying questions about the material.

---

### Output 6: Video Overview (Narrated Visual Deck)
- **NotebookLM Location:** Studio Panel -> Video Overview (or Chat prompt for narrated video storyboard).
- **Purpose:** Visual, slide-by-slide script and storyboard for a video explainer.
- **Prompt:**
```markdown
Generate a comprehensive 8-scene Video Script and Visual Storyboard for an educational YouTube video titled:
"From Zero to Transformer: Master Modern Data Science & AI with BSERC."

For each of the 8 scenes, provide:
1. Scene Title and Target Duration (seconds).
2. On-Screen Visuals: Exact description of diagrams, animations, Python code walk-throughs, or flowcharts displayed.
3. Spoken Voiceover Narration: Clear, engaging, pedagogically rigorous explanation grounded in the uploaded sources.
4. Key On-Screen Text / Callouts: Formulas, terminal commands, or code outputs.
5. Grounding Source Citation: Reference to the source grounding this scene.
```

---

### Output 7: Mind Map (Conceptual Hierarchy)
- **NotebookLM Location:** Studio Panel -> Mind Map (or Chat prompt generating a Mermaid syntax graph).
- **Purpose:** Structural visualization of how all 10 data science domains interconnect.
- **Prompt:**
```markdown
Create a comprehensive, structured Mind Map connecting the foundational concepts of Data Science, Classical ML, and Generative AI from the uploaded sources.

Provide the mind map in two formats:
1. A multi-level nested markdown outline illustrating the conceptual taxonomy (Parent -> Child -> Leaf concepts).
2. A valid, syntax-error-free Mermaid.js diagram (`graph TD`) mapping the flow from Raw Data Ingestion -> Wrangling -> Statistical Inference -> Classical Modeling -> Deep Transformers -> Production MLOps.

Ensure all relationship edges describe the functional link (e.g., "feeds into", "regularized by", "grounded by").
```

---

### Output 8: Infographic Blueprint
- **NotebookLM Location:** Studio Panel -> Infographic (or Chat prompt for single-page visual synthesis).
- **Purpose:** Compact, one-page visual layout blueprint ready for graphic design or digital publication.
- **Prompt:**
```markdown
Design a complete, high-impact one-page Infographic Blueprint titled:
"The Anatomy of Source-Grounded Data Science & AI."

Structure the blueprint into clear visual panels:
- Header Banner: Project mission, BSERC branding, and the three pillars (Code, Math, Domain Context).
- Panel 1: "The Pipeline" (Visual flow from raw telemetry to decision output).
- Panel 2: "The Engine Room" (NumPy SIMD contiguous memory vs. pure Python lists).
- Panel 3: "The Balance Scale" (Visual representation of the Bias-Variance trade-off with mathematical formulas).
- Panel 4: "The Transformer Engine" (Scaled dot-product attention breakdown).
- Panel 5: "The Real World" (Comparison of clean textbook data vs. noisy sensor telemetry).
- Footer: Key takeaways, verification badge, and primary source acknowledgments.

Provide exact copy, typography recommendations, color palette suggestions, and iconography ideas for each panel.
```

---

### Output 9: Slide Presentation Deck
- **NotebookLM Location:** Studio Panel -> Slide Deck (or Chat prompt for presentation slides).
- **Purpose:** 10-slide master presentation deck ready for academic lectures or technical reviews.
- **Prompt:**
```markdown
Generate an executive 10-slide presentation deck titled:
"Foundations and Frontiers of Data Science & Generative AI: A Source-Grounded Curriculum."

For each slide (Slides 1 to 10), format as follows:
- Slide Title & Subtitle.
- Bulleted Slide Body: 3–4 high-impact, punchy takeaways.
- Visual Layout & Graphic Recommendation: Specific chart, architecture diagram, or code callout box to display.
- Comprehensive Speaker Notes: 2–3 paragraphs of professional commentary explaining the concepts in depth with source citations.
- Key Discussion Question: One thought-provoking question to pose to the student audience.
```

---

### Output 10: Executive Briefing Document
- **NotebookLM Location:** Studio Panel -> Briefing Document / Report.
- **Purpose:** Formal executive technical brief for BSERC leadership and academic mentors.
- **Prompt:**
```markdown
Prepare a formal Executive Briefing Document titled:
"Curricular Synthesis & Source-Grounding Assessment for the BSERC Data Science Learning Assistant."

Structure the briefing into:
1. Executive Summary: Purpose, scope, and target learning outcomes.
2. Knowledge Corpus Analysis: Overview of the 10 uploaded primary sources across the 3 modalities (Documentation, Open Textbooks, University Lectures).
3. Pedagogical Architecture: How the curriculum bridges beginner-friendly onboarding with MS-level statistical and engineering rigor.
4. Source Verification & Trust Audit: Analysis of source-grounding fidelity, citation mechanisms, and mitigation of AI hallucination.
5. Operational Deployment Plan: Recommendations for student onboarding, interactive study workflows, and assessment tracking.
```

---

## 3. Step-by-Step Instructions to Test Every NotebookLM Feature

To satisfy the BSERC Week 1 project rubric, test every feature inside Google NotebookLM and log your observations:

### Feature 1: Multi-Modal Source Ingestion ("Feed It")
1. Navigate to [notebook.google.com](https://notebook.google.com) and click **New Notebook**.
2. **Add Web Sources:** Click **Add Source -> Web Link** and paste the URLs for Python, NumPy, Pandas, Scikit-learn, and Google MLCC.
3. **Add PDF Sources:** Click **Add Source -> Upload File** and upload the downloaded PDF versions of *ISLP* and *Mathematics for Machine Learning*.
4. **Add YouTube Sources:** Click **Add Source -> YouTube** and paste the video lecture URLs for MIT 6.0002 and MIT 6.S191.
5. **Verify Indexing:** Confirm all 10 sources show green checkmarks indicating full textual indexing.

### Feature 2: Sources On/Off Dynamic Toggling
1. In the left Sources panel, uncheck all sources except **SRC-02 (NumPy Guide)**.
2. Ask in chat: *"What is a DataFrame, and how does it handle missing values?"*
3. Observe that NotebookLM states it cannot answer because the active source only covers NumPy arrays, not Pandas DataFrames.
4. Check **SRC-03 (Pandas Guide)** back on and re-ask the question. Observe how the answer instantly populates with citations to Pandas documentation.

### Feature 3: Chat + Citation Verification ("Trust, But Check")
1. Ask in chat: *"What are the exact broadcasting rules in NumPy, and what happens when dimensions are unequal?"*
2. In the generated response, locate the small numeric bracket citations `[1]`, `[2]`.
3. Click directly on citation `[1]`. NotebookLM will immediately open the source panel and highlight the exact sentence from the NumPy documentation that substantiated the claim.

### Feature 4: Off-Topic Stress Test (Hallucination Defense)
1. Ask in chat an ungrounded or absurd question unrelated to data science:  
   *"Who won the 1982 FIFA World Cup, and what was the final score?"*
2. **Expected Behavior:** NotebookLM must explicitly reply that this information is not found in the uploaded sources, demonstrating strict grounding.

### Feature 5: Multilingual Output Generation ("Your Language")
1. Select one key concept (e.g., The Bias-Variance Trade-off) and prompt:  
   *"Explain the Bias-Variance trade-off using the uploaded sources, but write the entire explanation in Hindi [or your regional mother tongue: Telugu, Tamil, Marathi, Bengali, etc.], using clean technical vocabulary."*
2. Save the output to a note to prove cross-lingual grounding.

### Feature 6: Studio Panel Output Creation
1. Click through each icon in the **Studio Panel**:
   - Click **Audio Overview** -> Generate a Deep Dive podcast. Listen to 2 minutes, then click **Join In** to ask a spoken question.
   - Click **Study Guide** -> Generate and save.
   - Click **Flashcards** -> Test your recall on 5 cards.
   - Click **Quiz** -> Complete the 5-question interactive quiz.
   - Click **Mind Map** -> Inspect the generated concept graph.

### Feature 7: Share & Collaboration Link
1. In the top right corner, click the **Share** button.
2. Change permissions from *Restricted* to *"Anyone with the link can view"*.
3. Click **Copy link**.
4. Open an Incognito/Private browser window and paste the link to verify that an external reviewer or BSERC mentor can access your notebook without login friction.
