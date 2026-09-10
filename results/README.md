# Results

- `four-corpus.csv`: headline corpus, macro, and pooled WER.
- `corpus-scores.csv`: exact edit counts and corpus WER used for the headline table.
- `mega-asr-conditions.csv`: Mega-ASR acoustic-condition breakdown.
- `open-source-four-corpus.csv`: separate A5Sv2/open-source headline table. Every listed system
  has complete four-corpus coverage; partial and smoke-test runs are rejected.
- `open-source-corpus-scores.csv`: exact edit counts behind the open-source headline rows.

All values use word-onset v2 references (133,027 words) and the current 18,000-step A5Sv2 checkpoint.

All values come from one saved transcript run for every system and corpus. No CSV contains a
multi-run mean. Scores use `src/a5sv2_eval/score.py`; regenerate the SVGs with
`python tools/render_charts.py`. Confidence intervals are not reported.
