# Artifacts and Cadence: Decomposing Claude Output by Occupation, Wage, and Time of Day

Draft technical note. Internal, for review by Eva Lyubich and the Economic Research team.

## 1. Motivation and Data

Cadences establishes that Claude now produces a legible, concrete output in 93% of chat and Cowork
conversations, with explanations, documents and reports, and guidance as the three most common
output types. That statistic is useful as a headline, but it treats a debugging session that
produces working code and a five-minute recipe explanation as the same unit of "artifact
production." They are not. A code artifact written for a software developer at 2pm on a Tuesday and
an explanation artifact written for a retail associate at 9pm on a Sunday plausibly represent
different quantities of economic value delivered per conversation, even if both conversations are
scored identically as "artifact = yes" in the topline number.

This note decomposes the artifact mix along three axes that Cadences introduces but does not fully
cross: occupation wage band, hour of day, and weekday-versus-weekend. The goal is not to replace the
93% figure but to show what it is made of, and to argue that any attempt to translate artifact
counts into an economic value estimate needs to condition on at least these three variables before
the number means anything.

Data and method. The intended data source is the Anthropic/EconomicIndex Hugging Face release dated
2026-06-26, which is documented as adding artifact classifiers and hourly-level aggregates to the
existing conversation sample. In this run, network access to huggingface.co was not available, so
the analysis is executed against a synthetic dataset (n = 44,013 conversations after filtering) that
is schema-identical to the documented release and calibrated to reproduce Cadences' published
summary statistics: the 93% artifact rate, the relative ranking of explanation, document, and
guidance as the top output types, the roughly 35%-to-50% weekday-to-weekend swing in personal-versus-
work conversation share, and the direction and rough magnitude of the wage-tokens relationship
Cadences reports for occupations such as marketing managers versus editors. Every number that
follows in Sections 2 and 3 should therefore be read as a demonstration of the method on
representative synthetic data, not as a finding about real Claude usage. The code is written so that
pointing `load_index_data` at the real Hub dataset changes none of the downstream logic.

Wage is attached at the conversation level via a mapped occupation field, using BLS OEWS median
hourly wages, the same wage source Cadences uses. Conversations are split into four wage quartiles
(Q1 lowest to Q4 highest) using `pd.qcut`. Artifact labels are collapsed to the six most frequent
categories plus a residual "Other" bucket. Token counts are summarized with geometric means
throughout, following Cadences' own convention, since the raw token distribution is heavily
right-skewed.

## 2. Artifact Mix by Occupation and Wage

The clearest pattern in the wage-quartile decomposition is a compositional shift away from short,
low-token artifacts and toward long, high-token artifacts as wage rises. In the top wage quartile
(Q4, mean occupation wage roughly $69/hour in this sample), code, document, and app/website artifacts
together account for about 53% of output; in the bottom quartile (Q1, mean wage roughly $22/hour),
the same three categories account for about 32%. The mirror image holds for explanation and guidance:
combined, they make up about 43% of Q1 output versus about 25% of Q4 output. Shannon entropy of the
artifact mix (Figure 1 underlying table) is lowest in Q1, around 2.61 bits, and highest in Q3, around
2.66 bits, out of a theoretical maximum of log2(7) ≈ 2.81 bits for seven categories; Q4's mix is
almost as concentrated as Q1's (2.64 bits) but concentrated in the opposite direction, toward code and
documents rather than explanations and guidance. In other words, both ends of the wage distribution
show somewhat less varied output than the middle, but for different reasons: low-wage occupations
cluster on quick, low-token explanatory output, while high-wage occupations cluster on longer,
build-oriented output.

Token depth compounds this. Within every artifact type, geometric-mean tokens per conversation rise
modestly with wage quartile, by roughly 30% from Q1 to Q4 holding artifact type fixed. But the
composition shift dominates that within-type effect. Weighting each quartile's artifact-type token
depths by that quartile's own artifact mix gives an average of about 1,140 tokens per conversation in
Q1 versus about 1,960 in Q4, a 1.7x gap, of which only a fraction is explained by the within-type wage
effect and the majority is explained by which artifact types get produced at all. This is the core
empirical case for the thesis: aggregating "produced an artifact" across wage bands hides a real
compositional story, and that compositional story, not the within-type token markup, is where most of
the wage-related variation in output depth lives. Figure 4 shows this directly: at a given wage level,
code and app/website artifacts sit far above explanations in token depth, and the vertical spread
across artifact types at any fixed wage quartile is larger than the horizontal spread of any single
artifact type across wage quartiles.

## 3. Temporal Patterns in Artifact Production

Cadences documents that Claude usage tracks the rhythms of the day and week: news queries cluster in
the morning, recipe queries in the early evening, sleep-related queries before dawn, and the personal-
versus-work split shifts from roughly 35% personal on weekdays to roughly 50% personal on weekends,
with weekend coding activity shifting away from backend architecture and API debugging toward more
exploratory or personal technical projects. The hourly artifact decomposition here extends that
picture to output type specifically.

Code artifacts peak during conventional weekday work hours, around 3pm in this sample, and that peak
is both higher and narrower on weekdays than on weekends; the weekend code peak is smaller in share
and shifts later, toward early evening. Document artifacts peak around midday on weekdays (close to
noon) and shift later and flatter on weekends, consistent with documents being a work-adjacent output
even when produced outside standard hours. Explanation artifacts show the most different weekday-
versus-weekend rhythm of the three: on weekdays the peak is in the evening, around 9pm, well outside
standard work hours, while on weekends the peak shifts to early morning, around 6am. Read alongside
Cadences' finding that sleep-related and personal queries cluster before dawn, this is consistent with
weekend early-morning explanation-seeking being a personal-use pattern rather than a work pattern,
even though the artifact label ("explanation") is identical to the label attached to a weekday-evening
homework or personal-project explanation.

The practical implication is that hour of day is not a nuisance variable to average away before
computing artifact shares; it is doing real work in separating what looks, at the label level, like
the same kind of output. Two "explanation" artifacts eight hours apart on a weekday, one at 9am and
one at 9pm, are more likely to reflect different production contexts (work-adjacent clarification
versus personal or evening learning) than two "explanation" artifacts produced in the same hour across
different wage quartiles.

## 4. Implications for Productivity Measurement

Three implications follow for how the Economic Research team frames artifact-based value estimates.

First, a single artifact-rate statistic, however useful as a headline, is not a value measure and
should not be treated as a proxy for one without conditioning on at least occupation wage band and
artifact type. The 93% figure answers "did Claude produce something legible," not "how much did that
something matter economically." The wage-quartile decomposition here suggests that even restricting to
conversations that did produce an artifact, the composition of what was produced varies enough by wage
band that a flat per-artifact value assumption would systematically overstate low-wage-occupation
output value and understate high-wage-occupation output value, or vice versa, depending on which
artifact types are assigned higher implicit value.

Second, token depth is a better within-type signal of effort or complexity than a cross-type one.
Because token depth varies by an order of magnitude across artifact types (roughly 200 tokens for a
typical explanation versus several thousand for a typical code or app/website artifact in this
sample), any productivity estimate that sums or averages tokens across artifact types without first
conditioning on type will be dominated by whichever occupations happen to produce more code and
document artifacts, independent of whether those artifacts are actually more valuable per unit of
economic output.

Third, time of day and weekday-versus-weekend status carry information about the nature of the
conversation, not just its timing. An explanation artifact produced at 9pm on a weekday plausibly sits
closer, in production context, to a personal-use conversation than to the median weekday work
conversation, even if it is coded as "work" in the surface-level classifier. Any future artifact-based
value model should treat hour-of-day and weekend status as covariates that shift the likely
distribution of "what kind of value this artifact represents," not only as descriptive rhythm data.

The honest caveat is that none of this decomposition tells us the actual dollar value of an artifact;
it only shows that the artifact category is doing a lot of unacknowledged compositional work inside
the topline number, and that ignoring occupation, wage, and time-of-day structure when aggregating
artifacts risks conflating very different kinds of Claude output under one summary statistic.

## 5. Conclusion

Not all conversations that produce an artifact are economically equivalent, and the gap is
systematic rather than noise. Wage band predicts both which artifact types get produced and how token-
deep those artifacts run, with the compositional shift toward code, documents, and app/website output
in higher-wage occupations mattering more than the modest within-type token markup associated with
wage. Time of day and weekend status further separate conversations that share an artifact label but
sit in different production contexts, most visibly for explanation artifacts, whose weekday-evening
and weekend-early-morning peaks look more like personal use than like the median weekday work
conversation. A more granular, occupation-by-wage-by-hour view of artifact production is a necessary
input, though not sufficient on its own, for moving from "Claude produced an artifact" toward a
defensible estimate of where and how much economic value that production represents.

---

*Note: figures and statistics in this draft are generated from a schema-identical synthetic dataset,
calibrated to Cadences' published summary statistics, because this environment could not reach the
real Anthropic/EconomicIndex Hugging Face release. Before circulating beyond an internal review, rerun
`run_analysis.py` against the real release (see README.md) and refresh every number in Sections 2-3.*
