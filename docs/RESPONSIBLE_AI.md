# Responsible-AI analysis

The Research Discovery & Collaboration Engine is a decision-support component
inside the AI University architecture. It is not a student-risk classifier and
must not be presented as an automatic case-management or faculty-assignment
system.

## Intended use

The service ranks publication records that are semantically related to a
research topic. A human can use the ranked list and paper evidence to begin a
research-support conversation. The service is appropriate for demonstration,
research discovery, and staff-assisted exploration.

It is not appropriate to:

- grade, discipline, or rank students;
- determine academic eligibility or access to services;
- automatically contact a student or route a case;
- claim that a researcher is confirmed faculty, course instructor, or mentor;
- treat a confidence band as a calibrated probability of correctness.

## Cross-project privacy boundary

In the Group 6 → Group 7 workflow, the bridge sends only a resolved topic such
as `natural language processing`. Student message text, identifiers, S3T
confidence, crisis details, and human-review reasons remain in Group 6.
This code path has no application-level request persistence, and its error
response intentionally avoids returning internal exception details. Deployment
logs and infrastructure retention still need explicit University policy.

The default ranking strategy is a publication-similarity heuristic with an
author-count discount. Publication year is displayed as evidence but is not a
recency weight. The result is not a measure of unique expertise, teaching
status, faculty appointment, or mentor availability.

Before any real deployment, the University architecture still needs explicit
policies for consent, retention, access control, auditability, student notice,
and a process to contest or correct a generated recommendation.

## Explainability

Every recommendation contains:

1. a rank and similarity score;
2. a confidence band with an explanation and score margin;
3. the number of matched papers; and
4. paper-level evidence with title, publication year, and similarity.

These fields explain why a result was retrieved. They do not prove that the
researcher is the best human contact or that the evidence is free of gaps.
Reviewers should inspect the papers and use institutional knowledge before
taking action.

## Known risks and mitigations

| Risk | Why it matters | Mitigation in this project |
|---|---|---|
| Publication-coverage bias | OpenAlex records may overrepresent some fields, venues, or authors. | Show evidence and source metadata; describe results as publication matches. |
| Faculty/mentor overclaim | A prolific author may not teach at the institution or be available. | UI and docs explicitly avoid “confirmed faculty” claims. |
| Confidence overinterpretation | Similarity-derived confidence is not a probability. | Return a disclaimer in model metadata and require human review. |
| Co-author concentration | A single paper can make many co-authors look relevant. | Use the Week 5 author-discount strategy and expose matched-paper evidence. |
| Query ambiguity | “Transformer” or broad course names can retrieve mixed topics. | Allow strategy selection, show evidence, and require reviewer interpretation. |
| Stale catalog mapping | A placeholder course catalog can resolve a code to the wrong topic. | Show the resolution source in the Group 6 UI and require catalog maintenance. |
| Sensitive-data leakage | Sending the original message would exceed the Group 7 purpose. | Enforce the topic-only bridge contract and avoid query text in API errors. |
| Automation bias | A polished ranked list can be mistaken for a decision. | Keep advisory-only language visible in the UI, API metadata, and runbook. |

## Evaluation caveats

The available evaluation data and labels are project artifacts, not evidence of
real-world student or faculty outcomes. Benchmark scores should be used for
engineering comparisons only. They should not be reported as
validated institutional performance. Real deployment would require a reviewed
data-collection plan, independent labels, subgroup analysis, monitoring, and a
rollback path.

## Demo safety statement

The class demonstration must state:

- a staff member reviews every result;
- no action is taken automatically;
- Group 6 training data are generated messages and real-student performance is unknown;
- Group 6 keyword signals can miss crisis language or mishandle negation;
- Group 7 returns publication-based researcher matches, not a verified teaching roster;
- confidence labels are prioritization aids, not calibrated probabilities.
