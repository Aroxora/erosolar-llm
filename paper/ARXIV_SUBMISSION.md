# arXiv Submission Instructions

## Author: Bo Shang <bo@shang.software>

---

## Step 1: Register arXiv Account

1. Go to https://arxiv.org/user/register

2. Fill in registration form:
   - **Email:** `bo@shang.software`
   - **Username:** Choose (e.g., `boshang` or `boshang_erosolar`)
   - **Password:** Choose a strong password

3. Verify email - arXiv will send confirmation to `bo@shang.software`

4. Complete profile:
   - **Name:** Bo Shang
   - **Affiliation:** Erosolar AI Research
   - **Country:** [Your country]

5. **IMPORTANT:** First-time submitters need endorsement for cs.LG/cs.AI categories
   - Request endorsement at: https://arxiv.org/auth/endorse
   - Or get endorsed by an existing arXiv author

---

## Step 2: Prepare Submission

### Build the paper:
```bash
cd paper/
make all          # Compile PDF
make arxiv        # Create submission package
```

### Files included:
- `main.tex` - LaTeX source
- `arxiv_submission.tar.gz` - Ready to upload

---

## Step 3: Submit to arXiv

1. Log in at https://arxiv.org/

2. Click "Submit" → "Start New Submission"

3. Select categories:
   - **Primary:** cs.LG (Machine Learning)
   - **Secondary:** cs.AI (Artificial Intelligence), cs.CL (Computation and Language)

4. Upload `arxiv_submission.tar.gz`

5. Fill metadata:
   ```
   Title: Provably Correct Generative Adversarial Training:
          Grounded Verification for Reliable Training-Data Generation

   Authors: Bo Shang

   Abstract: We present a verification framework for generating
   training data whose members are checked for correctness before
   use. Unlike traditional Generative Adversarial Networks (GANs),
   which employ neural discriminators susceptible to the same
   hallucination failures as generators, our approach replaces
   learned discrimination with grounded verification—deterministic
   methods including code execution, symbolic mathematics, and
   authoritative database queries. We prove that, under the stated
   independence and false-positive-rate assumptions, training
   candidates passing our verification pipeline are correct with
   probability ≥ 1 - ε, where ε → 0 as verification methods increase.
   We further state a Capability Amplification argument: a model
   trained exclusively on verified data avoids its teacher's verified
   errors, suggesting improved reliability on verifiable tasks.
   Scope note: this paper contributes the framework and its formal
   analysis; the quantitative figures in the experiments section are
   illustrative projections, not measured results. No large-scale
   empirical evaluation has been conducted for this version, and no
   performance claim should be read as benchmarked.

   Comments: 15 pages, 3 tables, code available at
             https://github.com/ErosolarAI/erosolar-llm

   ACM-class: I.2.7; I.2.6

   MSC-class: 68T05
   ```

6. Preview and submit

7. Paper will be available within 1-2 days at:
   `https://arxiv.org/abs/26XX.XXXXX`

---

## Step 4: After Submission

### Get DOI:
arXiv papers automatically get a DOI in format:
`10.48550/arXiv.26XX.XXXXX`

### Update paper:
1. Go to "User" → "My Submissions"
2. Click "Replace" on the submission
3. Upload new `arxiv_submission.tar.gz`

### Cross-post:
Consider also posting to:
- **Papers With Code:** https://paperswithcode.com/ (link to GitHub repo)
- **Semantic Scholar:** Auto-indexed from arXiv
- **Google Scholar:** Auto-indexed from arXiv

---

## Alternative: OpenReview / HuggingFace Papers

If arXiv endorsement is delayed:

### HuggingFace Papers
1. https://huggingface.co/papers
2. Upload PDF directly
3. Link to model/dataset on HuggingFace

### OpenReview
1. https://openreview.net/
2. Submit as preprint
3. No endorsement required

---

## Citation (BibTeX)

Once published, cite as:

```bibtex
@article{shang2026pcgat,
  title={Provably Correct Generative Adversarial Training:
         Grounded Verification for Reliable Training-Data Generation},
  author={Shang, Bo},
  journal={arXiv preprint arXiv:26XX.XXXXX},
  year={2026}
}
```

---

## Contact

- **Author:** Bo Shang
- **Email:** bo@shang.software
- **Repository:** https://github.com/ErosolarAI/erosolar-llm
