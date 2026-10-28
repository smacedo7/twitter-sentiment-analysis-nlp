# Error analysis

The best model misclassified **61,651 of 318,134** test tweets (accuracy 80.6%).

## Type of mistake

- Negative tweets predicted as Positive: **32,956**
- Positive tweets predicted as Negative: **28,695**

## Error rate by tweet length (words after cleaning)

| Words | Tweets | Error rate |
|---|---:|---:|
| 1-3 | 70,630 | 16.8% |
| 4-7 | 127,060 | 18.3% |
| 8-12 | 96,962 | 21.6% |
| 13+ | 23,482 | 23.8% |

## Error rate by model confidence

| Confidence | Tweets | Error rate |
|---|---:|---:|
| 50-60% | 44,055 | 44.2% |
| 60-70% | 48,576 | 32.9% |
| 70-80% | 55,851 | 22.1% |
| 80-90% | 70,470 | 12.7% |
| 90-100% | 99,182 | 4.9% |

## Words most associated with mistakes

Words that appear in at least 20 test tweets, ordered by error rate.

| Word | Tweets | Error rate |
|---|---:|---:|
| noob | 22 | 54.5% |
| invest | 26 | 53.8% |
| rp | 23 | 52.2% |
| drown | 23 | 52.2% |
| chars | 23 | 52.2% |
| eng | 20 | 50.0% |
| confuse | 30 | 50.0% |
| sans | 28 | 50.0% |
| tsk | 42 | 50.0% |
| creep | 22 | 50.0% |
| clash | 27 | 48.1% |
| property | 27 | 48.1% |
| nursery | 25 | 48.0% |
| embarrassed | 25 | 48.0% |
| hoe | 23 | 47.8% |

## Most confident mistakes

Many of these are labeling noise or sarcasm: Sentiment140 labels come from emoticons, not from people.

| Tweet | Actual | Predicted | Confidence |
|---|---|---|---:|
| @user That is sad!  | Positive | Negative | 100% |
| @user We miss you Noods  | Positive | Negative | 100% |
| @user and i will miss u  | Positive | Negative | 100% |
| miss you @user  | Positive | Negative | 100% |
| @user will miss you too  | Positive | Negative | 100% |
| i miss my him!!!!  | Positive | Negative | 100% |
| @user I miss you in the greenroom.   | Positive | Negative | 100% |
| @user I miss you too ! &amp;&amp; kkay  | Positive | Negative | 100% |
| @user I miss you bunchessss!  | Positive | Negative | 100% |
| @user I miss you both so much  | Positive | Negative | 100% |
