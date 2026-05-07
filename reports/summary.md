# World Cup 2026 Prediction Summary

Simulations run: 10,000
Model type: logistic
Historical data cutoff: 2026-05-04

## Model Evaluation

- Accuracy: 0.599
- Log loss: 0.899
- Brier score, home-win class: 0.187

## Top Winner Probabilities

| Rank | Team | Winner probability |
|---:|---|---:|
| 1 | Spain | 16.44% |
| 2 | Argentina | 13.74% |
| 3 | France | 10.43% |
| 4 | Brazil | 7.27% |
| 5 | England | 6.54% |
| 6 | Ecuador | 6.42% |
| 7 | Colombia | 5.66% |
| 8 | Portugal | 4.25% |
| 9 | Japan | 3.71% |
| 10 | Netherlands | 3.04% |

## Notes

- Probabilities are model estimates, not certainties.
- Current features include Elo strength and opponent-adjusted recent form, so big wins over weak teams are capped and discounted.
- The default knockout simulation reseeds qualified teams and should be replaced with official fixture mapping for final publication.
- Refresh the group/team CSV before publishing if FIFA updates team names, groups, or fixtures.