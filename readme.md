# Employee Sentiment & Culture Analysis — UK Glassdoor Reviews

**Predicting employee advocacy from multi-dimensional review data.**
A People Analytics project across ~838,000 UK Glassdoor reviews (2008–2021).

*Zikra Aditia — People Analytics & Data Science*

---

## The question

What actually makes an employee recommend their workplace — or refuse to? And once
we know, where should HR act first for the biggest return?

Low advocacy isn't just a reputation problem. Gallup puts the cost of one disengaged
employee at roughly 34% of their salary. For a 1,000-person organization, the
employees who wouldn't recommend represent an estimated **£1.15M–£2.69M** in annual
cost exposure.

## What I found

- **Culture is the strongest driver of advocacy** — confirmed by two independent
  methods (a chi-square read and a controlled model, odds ratio 2.46). Senior
  Management and Career Opportunities follow close behind.
- **Compensation is a secondary, hygiene-type factor.** Pay matters, but it isn't the
  lever. This redirects HR spend away from blanket pay rises toward culture and
  leadership.
- **Culture and leadership compound.** When both are strong, 98% of employees
  recommend; when both are weak, only 12% do. Where both are weak, they have to be
  fixed together, not one at a time.
- **The text backs the numbers.** Detractors mention management 2.5× more than
  promoters. Promoters complain about long hours *more* than detractors — yet still
  recommend. Rough conditions are tolerated; bad management is the dealbreaker.
- **Advocacy is highly predictable** from the sub-ratings (recall 0.86 on detractors),
  which drives a well-calibrated High / Medium / Low risk-tiering system for HR.

## How it was built

`Data cleaning → EDA & driver analysis → predictive model → risk scoring → NLP
(voice of employee) → decision framework → cost-justified interventions →
experiment design → monitoring`

- **Model:** Balanced Logistic Regression, optimized for recall (missing a detractor
  is the costly mistake). Benchmarked against Random Forest — chosen for interpretable
  odds ratios a CHRO can follow, since the accuracy gap was ~0.1%.
- **NLP:** BERTopic was tried and rejected (short, generic review text produced
  incoherent topics); targeted n-gram analysis gave clear, interpretable themes.
  Method chosen to fit the data, not to look impressive.

## An honest note on the limits

Every claim here is **association, not causation**. The model shows *where* advocacy
and the ratings move together; it can't prove that improving a driver *causes*
advocacy to rise. A randomized pilot (described in the notebook) would be needed to
prove that. The `recommend` field is used as a **proxy** for advocacy — the data has
no attrition or salary columns. These limits are stated openly throughout; that
honesty is the point.

## Repository contents

| File | What it is |
|---|---|
| `glassdoor_sentiment.ipynb` | The full analysis notebook, end to end |
| `dashboard.py` | Interactive Streamlit dashboard (Executive + Risk & Voice views) |
| `requirements.txt` | Dependencies for the dashboard |
| `deck/` | CEO/CHRO presentation deck (14 slides) |
| `docs/` | Interview cheatsheet, methodology library, writing style guide, ethics notes |

## Running the dashboard

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

Opens at `http://localhost:8501`. The dashboard is self-contained — the model is
reconstructed from the notebook's coefficients, so no external data file is needed.

*Note: the raw review dataset is not included in this repo (it is large and publicly
available on Kaggle). The analysis notebook documents the full cleaning pipeline.*

---

*Part of a People Analytics portfolio. See also: IBM HR Attrition Analysis.*
