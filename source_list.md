# Source Collection & Grounding Repository
## Project 1: Data Science & AI Knowledge Notebook – A Source-Grounded Learning Assistant
**Organization:** Bharat Space Education Research Centre (BSERC) – GenAI Internship  
**Track:** Student / MS Data Science Learning Companion  
**Platform Target:** Google NotebookLM (`notebook.google.com`)  

---

## 1. Executive Overview

This repository document catalogues the **10 verified, primary learning sources** curated to build a trustworthy, source-grounded Data Science & Artificial Intelligence knowledge base in Google NotebookLM.

To strictly adhere to the project standards of the **Bharat Space Education Research Centre (BSERC)**, every source selected meets three foundational criteria:
1. **Verifiability:** Zero hallucinations, invented citations, or dead links. Every URL points to actively maintained official documentation, legally distributed academic textbooks, or university lecture repositories.
2. **Pedagogical Diversity:** The collection spans **three distinct modalities**:
   - Official Documentation & Industry Standards (5 sources)
   - Peer-Reviewed Open-Access Textbooks & Academic PDFs (3 sources)
   - University-Level Video Lectures & OpenCourseWare (2 sources)
3. **Curricular Coverage:** Seamlessly covers the 10 core domains requested by the BSERC curriculum:
   - Fundamentals of Data Science
   - Python for Data Science
   - NumPy & Pandas
   - Data Cleaning & Preprocessing
   - Data Visualization
   - Statistics & Exploratory Data Analysis (EDA)
   - Machine Learning Fundamentals
   - Generative AI & Large Language Models (LLMs)
   - Real-World Applications of AI & Data Science
   - Career Opportunities in Data Science

---

## 2. Source Master Index

| ID | Source Title | Author / Institution | Modality | Primary Topics Covered | Direct URL |
|---|---|---|---|---|---|
| **SRC-01** | *The Python Tutorial (Python 3.12+)* | Python Software Foundation (PSF) | Official Documentation / Web | Python for Data Science, Control Flow, Data Structures, OOP | [docs.python.org/3/tutorial/](https://docs.python.org/3/tutorial/) |
| **SRC-02** | *NumPy User Guide & Fundamentals* | NumPy Developers / NumFOCUS | Official Documentation / Web | Array Computing, Vectorization, Broadcasting, Linear Algebra | [numpy.org/doc/stable/user/](https://numpy.org/doc/stable/user/) |
| **SRC-03** | *Pandas User Guide* | The Pandas Development Team / NumFOCUS | Official Documentation / Web | DataFrames, Series, Data Cleaning, Missing Data, GroupBy, Time Series | [pandas.pydata.org/docs/user_guide/](https://pandas.pydata.org/docs/user_guide/) |
| **SRC-04** | *Scikit-learn User Guide: Machine Learning in Python* | Scikit-learn Consortium / Inria | Official Documentation / Web | Supervised/Unsupervised ML, Preprocessing, Model Evaluation, Pipelines | [scikit-learn.org/stable/user_guide.html](https://scikit-learn.org/stable/user_guide.html) |
| **SRC-05** | *Google Machine Learning Crash Course (MLCC)* | Google for Developers / Google AI | Interactive Course / Docs | ML Fundamentals, Loss, Gradient Descent, Overfitting, Fairness, GenAI | [developers.google.com/machine-learning/crash-course](https://developers.google.com/machine-learning/crash-course) |
| **SRC-06** | *An Introduction to Statistical Learning: with Applications in Python (ISLP)* | Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, Jonathan Taylor | Educational Textbook (Free Open PDF) | Statistics, Regression, Classification, Resampling, Tree Models, EDA | [statlearning.com](https://www.statlearning.com) |
| **SRC-07** | *Mathematics for Machine Learning* | Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong (Cambridge Univ. Press) | Educational Textbook (Open Access PDF) | Linear Algebra, Analytic Geometry, Matrix Decompositions, Probability | [mml-book.github.io/book/mml-book.pdf](https://mml-book.github.io/book/mml-book.pdf) |
| **SRC-08** | *Python Data Science Handbook: Essential Tools for Working with Data* | Jake VanderPlas (O'Reilly Open Edition) | Educational Textbook / Web Notebooks | NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn Workflow | [jakevdp.github.io/PythonDataScienceHandbook/](https://jakevdp.github.io/PythonDataScienceHandbook/) |
| **SRC-09** | *MIT 6.0002 Introduction to Computational Thinking and Data Science* | Prof. John Guttag, MIT OpenCourseWare | Educational Video Series / Courseware | Computational Thinking, Simulation, Sampling, Statistics, Optimization | [ocw.mit.edu/courses/6-0002...](https://ocw.mit.edu/courses/6-0002-introduction-to-computational-thinking-and-data-science-fall-2016/) & [YouTube Playlist](https://www.youtube.com/playlist?list=PLUl4u3cNGP62UGJSR4t7jM6eX31572X2x) |
| **SRC-10** | *MIT 6.S191 Introduction to Deep Learning* | Alexander Amini & Ava Amini, MIT | Educational Video Series / Course Portal | Deep Learning, Neural Networks, Transformers, Generative AI, Computer Vision | [introtodeeplearning.com](https://introtodeeplearning.com/) & [YouTube Channel](https://www.youtube.com/@AAmini/videos) |

---

## 3. Detailed Source Profiles & Grounding Metadata

### Source 1: The Python Tutorial
- **Publisher / Author:** Python Software Foundation (PSF) – Primary Author: Guido van Rossum & Community Contributors.
- **Source Type:** Official Documentation & Interactive Web Guide.
- **Direct URL:** [https://docs.python.org/3/tutorial/](https://docs.python.org/3/tutorial/)
- **Topics Covered:** Python execution model, primitive data types (integers, floats, strings), compound data structures (lists, tuples, dictionaries, sets), control flow (`if`, `for`, `while`, `match-case`), functions, lambda expressions, modules, exceptions, classes, and standard library tours.
- **Why Trustworthy:** The PSF is the official governing body of the Python language. This documentation is the canonical, primary-source definition of Python syntax, memory behavior, standard libraries, and language specifications.
- **NotebookLM Ingestion Instructions:**
  - Add via NotebookLM *Web Link* modal using `https://docs.python.org/3/tutorial/`.
  - Alternatively, export selected core chapters (e.g., Section 3 "An Informal Introduction to Python", Section 4 "More Control Flow Tools", and Section 5 "Data Structures") as clean PDF/text files for focused ingestion.

---

### Source 2: NumPy User Guide & Fundamentals
- **Publisher / Author:** NumPy Developers / NumFOCUS.
- **Source Type:** Official Documentation & Scientific Software Reference.
- **Direct URL:** [https://numpy.org/doc/stable/user/](https://numpy.org/doc/stable/user/)
- **Topics Covered:** N-dimensional array object (`ndarray`), contiguous memory layout, vectorized arithmetic, strides, axis semantics, broadcasting rules, indexing and slicing, linear algebra routines (`numpy.linalg`), random number generation, and universal functions (`ufuncs`).
- **Why Trustworthy:** NumPy is the foundational library underlying virtually every numerical and scientific Python package (Pandas, SciPy, Scikit-learn, PyTorch, TensorFlow). Its user guide is authored and reviewed by core scientific Python maintainers.
- **NotebookLM Ingestion Instructions:**
  - Add the specific fundamental user guide chapters: `https://numpy.org/doc/stable/user/absolute_beginners.html` and `https://numpy.org/doc/stable/user/basics.broadcasting.html` via Web Link.
  - Or download the NumPy stable reference documentation PDF from the official NumPy releases page and upload it directly.

---

### Source 3: Pandas User Guide
- **Publisher / Author:** The Pandas Development Team / NumFOCUS (Originally authored by Wes McKinney).
- **Source Type:** Official Documentation & API Guide.
- **Direct URL:** [https://pandas.pydata.org/docs/user_guide/](https://pandas.pydata.org/docs/user_guide/)
- **Topics Covered:** `Series` and `DataFrame` data structures, tabular indexing, label-based vs. integer-position selection (`loc` vs. `iloc`), missing data representation and imputation (`NaN`, `NA`), merge/join/concatenate operations, split-apply-combine (`groupby`), categorical data, and time series handling.
- **Why Trustworthy:** Pandas is the worldwide industry benchmark for data manipulation, cleaning, and preprocessing in Python. Its documentation provides precise explanations of edge-case behaviors, copy-versus-view semantics, and data alignment.
- **NotebookLM Ingestion Instructions:**
  - Add core pages via NotebookLM Web Link: "10 Minutes to pandas", "Essential basic functionality", and "Working with missing data".
  - For comprehensive offline ingestion, download the official `pandas.pdf` manual directly from PyData documentation archives.

---

### Source 4: Scikit-learn User Guide: Machine Learning in Python
- **Publisher / Author:** Scikit-learn Consortium / Inria / NumFOCUS (Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, et al.).
- **Source Type:** Official Documentation, Algorithmic Reference & Mathematical Guide.
- **Direct URL:** [https://scikit-learn.org/stable/user_guide.html](https://scikit-learn.org/stable/user_guide.html)
- **Topics Covered:** Supervised learning algorithms (linear models, logistic regression, support vector machines, decision trees, ensemble methods like Random Forest and Gradient Boosting), unsupervised learning (k-means, PCA), model evaluation (confusion matrix, ROC-AUC, precision-recall, cross-validation), preprocessing transformers (`StandardScaler`, `OneHotEncoder`), and composite `Pipeline` architectures.
- **Why Trustworthy:** Scikit-learn is the golden standard for production and academic classical machine learning. The User Guide not only documents APIs but provides mathematical formulations, loss function derivations, and trade-offs for every algorithm.
- **NotebookLM Ingestion Instructions:**
  - Ingest targeted user guide sections (e.g., Chapter 1: Supervised models, Chapter 3: Model selection and evaluation, Chapter 6: Dataset transformations) via URL or print-to-PDF upload.

---

### Source 5: Google Machine Learning Crash Course (MLCC)
- **Publisher / Author:** Google for Developers / Google AI Research Team.
- **Source Type:** Industry Interactive Course & Technical Curriculum.
- **Direct URL:** [https://developers.google.com/machine-learning/crash-course](https://developers.google.com/machine-learning/crash-course)
- **Topics Covered:** Framing machine learning problems, loss functions (MSE, LogLoss), gradient descent variants (SGD, mini-batch), model representation, feature engineering, regularizations (L1, L2), classification thresholds, fairness in AI, neural network building blocks, embeddings, and introductions to Large Language Models.
- **Why Trustworthy:** Developed by Google's premier AI researchers and engineers to train Google staff and the global developer community. It reflects modern best practices applied at hyperscale.
- **NotebookLM Ingestion Instructions:**
  - Paste modular lesson URLs into NotebookLM's Web Source modal (e.g., Framing, Descending into ML, Reducing Loss, Generalization, Representation).

---

### Source 6: An Introduction to Statistical Learning: with Applications in Python (ISLP)
- **Publisher / Author:** Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, and Jonathan Taylor. Published by Springer Texts in Statistics (2023).
- **Source Type:** Academic Peer-Reviewed Textbook (Legally Free Open-Access PDF).
- **Direct URL:** [https://www.statlearning.com](https://www.statlearning.com) (Companion code & resources: [https://www.statlearning.com/isl-python](https://www.statlearning.com/isl-python)).
- **Topics Covered:** Statistical learning theory, bias-variance trade-off, multiple linear regression, hypothesis testing ($p$-values, $F$-statistics), logistic regression, linear discriminant analysis (LDA), resampling methods (cross-validation, bootstrapping), linear model selection and regularization (ridge regression, lasso), tree-based methods (bagging, random forests, boosting), and unsupervised learning (PCA, clustering).
- **Why Trustworthy:** Authored by Stanford, USC, and University of Washington professors who are among the most cited statisticians in human history (Hastie and Tibshirani co-invented the Lasso and Generalized Additive Models). This is the definitive graduate/undergraduate textbook worldwide.
- **NotebookLM Ingestion Instructions:**
  - Download the official free legal PDF directly from [statlearning.com](https://www.statlearning.com) and upload the PDF file (or specific core chapters like Chapters 2, 3, 4, 5, 6, 8) into NotebookLM via the "Upload File" button.

---

### Source 7: Mathematics for Machine Learning
- **Publisher / Author:** Marc Peter Deisenroth (University College London), A. Aldo Faisal (Imperial College London), and Cheng Soon Ong (Australian National University). Published by Cambridge University Press (2020).
- **Source Type:** Academic Peer-Reviewed Textbook (Legally Free Open-Access PDF).
- **Direct URL:** [https://mml-book.github.io/book/mml-book.pdf](https://mml-book.github.io/book/mml-book.pdf) (Companion site: [https://mml-book.github.io/](https://mml-book.github.io/)).
- **Topics Covered:** Vector spaces, linear independence, basis, linear mappings, matrices, vector norms, inner product spaces, orthogonality, matrix decompositions (eigenvalues, eigenvectors, singular value decomposition - SVD), vector calculus (gradients, Jacobians, Hessians), probability distributions, continuous random variables, Bayes' theorem, maximum likelihood estimation (MLE), maximum a posteriori (MAP), and empirical risk minimization.
- **Why Trustworthy:** Published by Cambridge University Press and made freely accessible online by the authors. It bridges the gap between pure mathematics and applied machine learning algorithms.
- **NotebookLM Ingestion Instructions:**
  - Ingest the official PDF `mml-book.pdf` directly into NotebookLM. For targeted processing, extract Part I: "Mathematical Foundations" (Chapters 2, 5, 6) and upload.

---

### Source 8: Python Data Science Handbook: Essential Tools for Working with Data
- **Publisher / Author:** Jake VanderPlas. Published by O'Reilly Media (Second Edition open online release).
- **Source Type:** Educational Textbook & Open-Source Computational Notebook Collection.
- **Direct URL:** [https://jakevdp.github.io/PythonDataScienceHandbook/](https://jakevdp.github.io/PythonDataScienceHandbook/) (GitHub Repository: [https://github.com/jakevdp/PythonDataScienceHandbook](https://github.com/jakevdp/PythonDataScienceHandbook)).
- **Topics Covered:** IPython and Jupyter notebook ecosystems, NumPy array computation, Pandas data manipulation, Matplotlib and Seaborn visualization methodologies, and Scikit-learn end-to-end machine learning workflows.
- **Why Trustworthy:** Jake VanderPlas is a renowned astrophysicist and core open-source contributor who served on the scientific Python executive steering councils and works on the Google research team.
- **NotebookLM Ingestion Instructions:**
  - Ingest individual chapter web pages (e.g., Chapter 2: NumPy, Chapter 3: Pandas, Chapter 4: Matplotlib) via the Web URL option in NotebookLM.

---

### Source 9: MIT 6.0002 Introduction to Computational Thinking and Data Science
- **Publisher / Author:** Prof. John Guttag and Prof. Eric Grimson, Massachusetts Institute of Technology (MIT OpenCourseWare).
- **Source Type:** University Courseware & Educational Video Lecture Series.
- **Direct URL:** 
  - MIT OCW Course Page: [https://ocw.mit.edu/courses/6-0002-introduction-to-computational-thinking-and-data-science-fall-2016/](https://ocw.mit.edu/courses/6-0002-introduction-to-computational-thinking-and-data-science-fall-2016/)
  - Official YouTube Playlist: [https://www.youtube.com/playlist?list=PLUl4u3cNGP62UGJSR4t7jM6eX31572X2x](https://www.youtube.com/playlist?list=PLUl4u3cNGP62UGJSR4t7jM6eX31572X2x)
- **Topics Covered:** Computational modeling, optimization algorithms, dynamic programming, stochastic thinking, random walks, Monte Carlo simulations, sampling distributions, confidence intervals, experimental data interpretation, curve fitting, regression fallacy, and clustering.
- **Why Trustworthy:** MIT OpenCourseWare is the world standard for accessible elite academic curricula. Prof. John Guttag was the head of MIT's Department of Electrical Engineering and Computer Science (EECS) and a world authority on computer science education.
- **NotebookLM Ingestion Instructions:**
  - In NotebookLM, select **Add Source -> YouTube** and paste individual lecture URLs (such as Lecture 1: Optimization, Lecture 7: Confidence Intervals, Lecture 8: Monte Carlo Simulations, Lecture 10: Experimental Data). NotebookLM automatically parses transcripts into grounded source text.

---

### Source 10: MIT 6.S191 Introduction to Deep Learning
- **Publisher / Author:** Dr. Alexander Amini and Dr. Ava Amini, Massachusetts Institute of Technology (MIT EECS).
- **Source Type:** Advanced University Course Portal & Educational Video Lecture Series.
- **Direct URL:** 
  - Official Course Portal: [https://introtodeeplearning.com/](https://introtodeeplearning.com/)
  - Official YouTube Channel / Lectures: [https://www.youtube.com/@AAmini/videos](https://www.youtube.com/@AAmini/videos)
- **Topics Covered:** Perceptrons, multi-layer perceptrons, backpropagation, activation functions, convolutional neural networks (CNNs), recurrent neural networks (RNNs/LSTMs), Attention mechanisms, Transformers, Large Language Models (LLMs), Generative Adversarial Networks (GANs), Diffusion Models, reinforcement learning, and AI ethics/bias.
- **Why Trustworthy:** MIT 6.S191 is MIT's official introductory course on deep learning, updated annually with state-of-the-art research breakthroughs, sponsored by major AI research labs, and taught by leading MIT researchers.
- **NotebookLM Ingestion Instructions:**
  - Add via NotebookLM **YouTube URL** input (e.g., Lecture 1: Deep Learning Foundations, Lecture 2: Transformers and Attention, Lecture 5: Generative Models).
  - Also upload lecture slide PDFs available freely for download on [introtodeeplearning.com](https://introtodeeplearning.com/).

---

## 4. Source-to-Topic Grounding Coverage Matrix

| Topic Number & Name | Primary Sources Grounding This Topic | Supplementary Sources |
|---|---|---|
| **1. Fundamentals of Data Science** | SRC-05 (Google MLCC), SRC-09 (MIT 6.0002) | SRC-06 (ISLP Ch. 1-2), SRC-08 (VanderPlas) |
| **2. Python for Data Science** | SRC-01 (Python Official Docs), SRC-08 (VanderPlas Ch. 1) | SRC-09 (MIT 6.0002) |
| **3. NumPy and Pandas** | SRC-02 (NumPy Guide), SRC-03 (Pandas Guide) | SRC-08 (VanderPlas Ch. 2 & 3) |
| **4. Data Cleaning and Preprocessing** | SRC-03 (Pandas Guide), SRC-04 (Scikit-learn Guide Ch. 6) | SRC-05 (Google MLCC), SRC-06 (ISLP) |
| **5. Data Visualization** | SRC-08 (VanderPlas Ch. 4), SRC-09 (MIT 6.0002) | SRC-03 (Pandas Plotting) |
| **6. Statistics and Exploratory Data Analysis** | SRC-06 (ISLP Ch. 2-3), SRC-07 (MML Ch. 6), SRC-09 (MIT 6.0002) | SRC-08 (VanderPlas), SRC-05 (Google MLCC) |
| **7. Machine Learning Fundamentals** | SRC-04 (Scikit-learn), SRC-05 (Google MLCC), SRC-06 (ISLP) | SRC-07 (MML Ch. 9-12) |
| **8. Generative AI and Large Language Models** | SRC-10 (MIT 6.S191), SRC-05 (Google MLCC GenAI Modules) | SRC-04 (Scikit-learn Text Features) |
| **9. Real-World Applications of AI & DS** | SRC-05 (Google MLCC Case Studies), SRC-09 (MIT 6.0002) | SRC-10 (MIT 6.S191 Autonomous Systems/Biotech) |
| **10. Career Opportunities in Data Science** | SRC-05 (Google Developer Roles), Industry Standards (PSF/NumFOCUS) | SRC-06 (ISLP Academic & Applied Roles) |

---

## 5. Verification Checklist for NotebookLM Upload

When creating the notebook manually inside **Google NotebookLM** (`https://notebook.google.com`):
1. **Title the Notebook:** `Data Science & AI Knowledge Notebook – A Source-Grounded Learning Assistant`.
2. **Add Source Modality 1 (Web):**
   - Submit URLs for SRC-01, SRC-02, SRC-03, SRC-04, SRC-05, and SRC-08.
3. **Add Source Modality 2 (PDF / Textbooks):**
   - Download and upload PDF copies of *ISLP (Python edition)* (SRC-06) and *Mathematics for Machine Learning* (SRC-07).
4. **Add Source Modality 3 (YouTube / Video Lectures):**
   - Insert YouTube video links for MIT 6.0002 (SRC-09) and MIT 6.S191 (SRC-10).
5. **Verify Source Indexing:**
   - Ensure NotebookLM displays all 10 sources in the left Sources panel with active checkboxes.
   - Click each source to verify that character extraction and transcript parsing completed successfully without formatting errors.
