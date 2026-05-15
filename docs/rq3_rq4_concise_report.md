# MoltBook RQ3 and RQ4 Research Notes

## Research Question 3: Feature-Level Sentiment Patterns

RQ3 investigates whether sentiment in the MoltBook corpus is associated with observable textual, conversational, and user-level features. In other words, this question asks whether negative, neutral, and positive comments differ in measurable ways beyond the sentiment label itself.

The result is **partially supported**. Several features show statistically significant differences across sentiment classes, especially text length, upvotes, and exclamation use. However, the stronger assumption that deeper threads produce greater sentiment variability is not supported.

### Statistical Summary

The Kruskal-Wallis tests show that sentiment classes differ significantly in text length, thread depth, and upvotes. Text length in words is significant (`H = 19.73`, `p = 5.21e-05`), and text length in characters shows the same pattern (`H = 17.43`, `p = 1.64e-04`). Thread depth is statistically significant but weak (`H = 6.50`, `p = 0.0388`). Upvotes also differ across sentiment groups (`H = 14.67`, `p = 6.54e-04`).

Among binary features, exclamation marks are the clearest signal (`chi2 = 32.15`, `p = 1.05e-07`). Verification status is not significant (`p = 0.113`), and question marks are not significant (`p = 0.653`).

For sentiment variability, the relationship with thread depth is essentially absent (`r = -0.0096`, `p = 0.949`). By contrast, mean text length has a moderate positive association with sentiment variability (`r = 0.3935`, `p = 0.00621`). This means longer or more elaborated conversations tend to contain more variation in sentiment, while depth alone does not explain that variation.

### Figure 1: Feature Boxplots

![RQ3 feature boxplots](../data/figures/rq3_feature_boxplots_20260419T092811Z.png)

This figure compares feature distributions across negative, neutral, and positive sentiment classes.

The text-length panels show that neutral comments tend to be longer than both negative and positive comments. This is consistent with the interpretation that neutral comments often carry informational, explanatory, or analytical content. Positive comments are frequent but tend to be shorter on average, suggesting that supportive or affirmative discourse may require less elaboration.

Thread depth shows a small but statistically significant difference. Positive comments appear somewhat more often in deeper conversational positions, which may indicate that sustained interaction can become more constructive or affiliative. However, this effect is small and should not be overstated.

The upvote panel indicates that sentiment categories differ in engagement, although most comments receive very few upvotes. The polarity-compound panel separates the classes very strongly, but this should be interpreted cautiously because VADER compound score contributes directly to the sentiment labeling process.

### Figure 2: Sentiment Variability vs Thread Depth

![RQ3 variability scatter](../data/figures/rq3_variability_scatter_20260419T092811Z.png)

This figure plots thread-level sentiment variability against maximum thread depth. The fitted line is nearly flat, and the statistical result confirms that there is no meaningful relationship between depth and variability.

Professionally, this means that deeper threads are not automatically more emotionally diverse. Some shallow threads show high variability, and some deep threads remain relatively stable. Thread depth alone is therefore not a reliable predictor of sentiment complexity.

### Figure 3: Sentiment by Verification Status

![RQ3 verification status](../data/figures/rq3_verified_vs_unverified_20260419T092811Z.png)

This figure compares sentiment proportions between verified and unverified users.

Both groups show almost the same overall structure: neutral sentiment is dominant, positive sentiment is second, and negative sentiment is low. The statistical test confirms that verification status is not a strong sentiment predictor in this dataset.

The practical interpretation is that verified users do not appear to communicate in a substantially different emotional style from unverified users, at least under the current rule-based sentiment pipeline.

### RQ3 Conclusion

RQ3 shows that sentiment is connected to discourse features, but not in a simple way. Text length, upvotes, and exclamation marks matter; thread depth and verification status are weaker or non-significant. The most defensible conclusion is that sentiment variation in MoltBook is shaped more by communicative elaboration and expressive markers than by thread structure alone.

---

## Research Question 4: Robustness of Sentiment Findings

RQ4 examines whether the main sentiment conclusions remain stable when the pipeline changes. This is important because rule-based sentiment analysis can be sensitive to preprocessing choices and scoring tools.

The result is **partially supported**. The neutral-dominant conclusion is stable under basic cleaning and stricter filtering, but it changes when the scorer is replaced with VADER-only or SentiWordNet-only labels.

### Robustness Summary

The baseline ensemble result is neutral-dominant: negative `0.062`, neutral `0.548`, and positive `0.390`. The basic-clean variant produces the same result, and the strict-filter variant is almost identical: negative `0.063`, neutral `0.550`, and positive `0.386`.

The scorer-only variants behave differently. VADER-only produces a strongly positive corpus: negative `0.260`, neutral `0.038`, and positive `0.702`. SentiWordNet-only also makes positive the largest class: negative `0.127`, neutral `0.386`, and positive `0.487`.

This means the preprocessing choices are stable, but the sentiment scorer is a major source of sensitivity. The maximum positive-share deviation from baseline is `0.3117`, and the sensitive variants are `v3_vader_only` and `v4_swn_only`.

### Figure 4: Robustness Heatmap

![RQ4 robustness heatmap](../data/figures/rq4_robustness_heatmap_20260419T092811Z.png)

This heatmap shows the proportion of negative, neutral, and positive sentiment across five pipeline variants.

The baseline, basic-clean, and strict-filter rows are nearly identical. This supports the reliability of the main preprocessing pipeline: the conclusion is not simply caused by stopword removal, lemmatization, or minimum word-count filtering.

The VADER-only row is the main contrast. It produces a highly positive interpretation, with more than 70% of comments labeled positive. This suggests that VADER is more sensitive to positive lexical or surface-level affective cues in the MoltBook corpus.

The SentiWordNet-only row is less extreme than VADER but still shifts the dominant class to positive. This confirms that using only one lexicon can substantially change the research conclusion.

### RQ4 Conclusion

RQ4 shows that the main finding is robust to preprocessing but sensitive to model choice. Therefore, the ensemble method is more defensible than relying on either VADER or SentiWordNet alone.

The best academic interpretation is not that the corpus is neutral under every possible method. Rather, the correct claim is that the neutral-dominant conclusion is stable under cleaning and filtering variants when the ensemble method is used. Single-scorer variants produce more positive interpretations and should be treated as sensitivity checks rather than the primary result.

---

## Combined Interpretation

RQ3 and RQ4 together strengthen the overall NLP interpretation of the MoltBook corpus.

RQ3 shows that sentiment is related to measurable discourse features. Longer comments, engagement signals, and expressive punctuation are associated with sentiment differences. However, thread depth and verification status are not strong standalone predictors.

RQ4 shows that the main sentiment conclusion is methodologically stable under preprocessing changes but sensitive to the choice of sentiment scorer. This supports the use of the ensemble output as the primary result.

Overall, MoltBook discourse should be described as predominantly neutral and informational, with a meaningful positive component and relatively little negative sentiment. The evidence suggests a corpus shaped more by analytical and explanatory interaction than by emotional conflict.

