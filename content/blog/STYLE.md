# Notebook style rules

Every Notebook article, written by hand or drafted by the pipeline
(`scripts/notebook/`), follows these rules. The pipeline's validator checks
the ones marked [checked]; the rest are for the reviewer.

## The one rule

**No article without runnable code and an experiment with measured results.**
Long text-only essays are not published.

## Structure (in this order)

1. **Context:** what the problem is and why it matters, for a reader who is
   not a specialist.
2. **The question(s)** the article answers, stated before any method.
3. **Setup:** the data (where it comes from, exactly), the methods, and how
   they are measured.
4. **The code:** the important parts inline, the full script in
   `notebook-code/<slug>/`. [checked: at least one ```python block]
5. **Results:** tables and figures produced by the script.
6. **What the numbers say:** interpretation, including results that go
   against the expected story.
7. **What this experiment does not show** (limits). [checked: section present]
8. **Takeaways**, **Run it yourself**, **Sources**. [checked: Sources present]

## Numbers

- Every number about a result comes from running the script, and appears in
  its `results.json`. [checked: numbers in the prose are matched against
  results.json]
- No claim without a source. Papers are cited with authors, venue and year.
- If a result is surprising, investigate it before publishing (the bearing
  article found a dataset artifact this way).

## Code

- Runs on a normal CPU in under 10 minutes.
- Data: synthetic, a dataset bundled with SciPy or scikit-learn, or a small
  public file downloaded by the script. No accounts or keys.
- Fixed random seeds, so a rerun gives the same numbers.

## Language

- Plain words. Every technical term is explained the first time it appears.
- 1200-1800 words of prose (code not counted). [checked]
- Banned: "revolutionary", "game-changer", "groundbreaking", "paradigm shift",
  "cutting-edge", "state-of-the-art" as an adjective, "delve". [checked]
- No emoji.
