# Evaluation report

Hosted run with provider groq and search mode fixture. Results depend on the model and, when search is live, on the public web. Success rate ignores tasks that stayed rate-limited after retries; those are not model failures.

| Metric | Value |
| --- | --- |
| Mode | hosted |
| Provider | groq |
| Model | qwen/qwen3.8-27b |
| Search | fixture |
| Generated | 2026-11-08T03:56:06Z |
| Tasks | 15 |
| Passed | 14 |
| Rate limited | 0 |
| Scored | 15 |
| Success rate | 93.3% |
| Average steps | 2.73 |
| Average latency | 16658.89 ms |

| Category | Passed | Total |
| --- | --- | --- |
| math | 5 | 5 |
| code | 4 | 5 |
| factual | 5 | 5 |

| id | category | pass | steps | latency_ms | answer |
| --- | --- | --- | --- | --- | --- |
| math-multiply-add | math | yes | 3 | 735.96 | (17 × 23) + 5 = 396. |
| math-power | math | yes | 3 | 664.86 | 2 ** 10 = 1024. |
| math-sqrt | math | yes | 3 | 6031.77 | The square root of 144 is 12. |
| math-parentheses | math | yes | 3 | 22519.65 | The result is 26. |
| math-division | math | yes | 3 | 22460.02 | 100 / 4 + 7 = 32. |
| code-sum-squares | code | yes | 3 | 22197.37 | The sum of squares from 1 to 10 is 385. |
| code-fibonacci | code | no | 3 | 23611.08 | The 10th Fibonacci number (with the sequence starting 1, 1, 2, 3, 5, ..… |
| code-factorial | code | yes | 3 | 22412.91 | The factorial of 7 is 5040. |
| code-primes | code | yes | 3 | 26147.14 | There are 15 primes strictly below 50: 2, 3, 5, 7, 11, 13, 17, 19, 23, … |
| code-reverse | code | yes | 3 | 22309.44 | The reversed string is **citnega**. |
| fact-france-capital | factual | yes | 3 | 22377.89 | The capital of France is Paris. |
| fact-moon-landing | factual | yes | 2 | 14674.84 | 1969 — the first crewed Moon landing was Apollo 11, when Neil Armstrong… |
| fact-gold-symbol | factual | yes | 2 | 14538.07 | Au |
| fact-gatsby-author | factual | yes | 2 | 14555.98 | F. Scott Fitzgerald wrote *The Great Gatsby*, published in 1925. |
| fact-water-boiling | factual | yes | 2 | 14646.44 | The boiling point of water at sea level is 100 degrees Celsius (212 deg… |
