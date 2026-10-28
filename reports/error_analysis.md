# Error analysis

The best model misclassified **61,849 of 318,134** test tweets (accuracy 80.6%).

## Type of mistake

- Negative tweets predicted as Positive: **32,219**
- Positive tweets predicted as Negative: **29,630**

## The Neutral zone

The model answers **Neutral** for 17.2% of the test tweets. If it had been forced to choose a side for those tweets, it would have been wrong 42.7% of the time. On the other tweets, accuracy rises to **85.4%**.

## Error rate by tweet length (words after cleaning)

| Words | Tweets | Error rate |
|---|---:|---:|
| 1-3 | 70,630 | 16.8% |
| 4-7 | 127,060 | 18.4% |
| 8-12 | 96,962 | 21.7% |
| 13+ | 23,482 | 23.7% |

## Error rate by model confidence

| Confidence | Tweets | Error rate |
|---|---:|---:|
| 50-60% | 40,493 | 44.5% |
| 60-70% | 43,075 | 34.2% |
| 70-80% | 49,487 | 24.4% |
| 80-90% | 78,014 | 14.6% |
| 90-100% | 107,065 | 5.2% |

## Words most associated with mistakes

Words that appear in at least 20 test tweets, ordered by error rate.

| Word | Tweets | Error rate |
|---|---:|---:|
| noob | 22 | 54.5% |
| rp | 23 | 52.2% |
| chars | 23 | 52.2% |
| drown | 23 | 52.2% |
| clash | 27 | 51.9% |
| potty | 27 | 48.1% |
| nursery | 25 | 48.0% |
| hoe | 23 | 47.8% |
| tsk | 42 | 47.6% |
| betta | 45 | 46.7% |
| confuse | 30 | 46.7% |
| sans | 28 | 46.4% |
| invest | 26 | 46.2% |
| cleaner | 24 | 45.8% |
| kh | 24 | 45.8% |

## Most confident mistakes

Many of these are labeling noise or sarcasm: Sentiment140 labels come from emoticons, not from people.

| Tweet | Actual | Predicted | Confidence |
|---|---|---|---:|
| @user Hey say me something  haha now that you in love, love, love (8) forget about me? haha luv ya and im so happy because u happy | Negative | Positive | 99% |
| @user thanks. me too..  | Negative | Positive | 99% |
| @user  thanks. | Negative | Positive | 99% |
| @user thanks  | Negative | Positive | 99% |
| your so awesome  | Negative | Positive | 99% |
| @user thanks  | Negative | Positive | 99% |
| @user me 2  BUt thanks  | Negative | Positive | 99% |
| humf  thanks? | Negative | Positive | 99% |
| @user thank you  | Negative | Positive | 99% |
| @user   Less good. | Negative | Positive | 99% |
