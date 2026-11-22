# Pilot source review and execution

The assistant checked the ten draft answers against their cited extracted passages. All ten are supported paraphrases and retain speaker attribution. They are approved for a mechanics experiment, not a claim of dataset-wide quality. The original draft remains unchanged; reviewed copies are in `data/instructions/living_by_the_code/reviewed/`.

| Example suffix | Contributor | PDF pages | Review finding |
|---|---|---|---|
| 001 | Britt Barak | 23 | Leadership answer preserves planning, team autonomy and attention to detail. |
| 002 | Britt Barak | 24–25 | Gradual move into developer advocacy preserves the engineering foundation. |
| 003 | Britt Barak | 25–26 | Calendar and effort management reflect the passage. |
| 004 | Ellen Shapiro | 33 | Cross-platform learning does not imply equal expertise. |
| 005 | Ellen Shapiro | 35–36 | Teaching and mentoring deepen understanding; no extra factual claims. |
| 006 | Ellen Shapiro | 34–35 | Team support and diverse client work reflect her experience. |
| 007 | Lara Martin | 56 | Community participation and contribution are supported. |
| 008 | Lara Martin | 56–57 | Communication practice and learning through teaching are supported. |
| 009 | Lara Martin | 58 | Helping others grow and learning leadership are supported. |
| 010 | Antonio Leiva | 45 | Preserves uncertainty rather than presenting hindsight as prediction. |

Seven examples are training data; three Lara Martin examples are validation data. No final-test example is prepared or used by the pilot. Similar themes across interviews, especially teaching and community participation, limit how strongly these three validation examples can demonstrate generalization.

The CPU runner logs shifted response-only cross-entropy, supervised target counts, validation NLL weighted by target count, process RAM and elapsed time. It saves token inspections, adapter weights, dataset checksums, dependency versions and fixed greedy generations before and after training. End-of-answer tokens must remain supervised; prompt and padding labels are `-100`. The pilot refuses truncated examples and unreviewed drafts.

This small run tests the training mechanics. Lower loss or a changed answer does not establish useful improvement. Review generated answers for fidelity and attribution. Keep final-test sources reserved until experiment settings are fixed.
