"""
Adversarial Judge Agent - Harsh critic that reviews sector expert findings.

The judge:
1. Scrutinizes every investment thesis for logical fallacies
2. Challenges valuation assumptions and growth projections
3. Identifies confirmation bias and missing risks
4. Demands deeper evidence and contrarian perspectives
5. Scores research quality and sends feedback for iteration
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class JudgeCritique:
    sector: str
    agent_name: str
    timestamp: str
    overall_quality_score: float  # 0-100
    thesis_rigor_score: float  # 0-100
    risk_assessment_score: float  # 0-100
    data_quality_score: float  # 0-100
    contrarian_thinking_score: float  # 0-100
    actionability_score: float  # 0-100
    major_flaws: list[str] = field(default_factory=list)
    missing_analysis: list[str] = field(default_factory=list)
    logical_fallacies: list[str] = field(default_factory=list)
    biases_detected: list[str] = field(default_factory=list)
    questions_to_answer: list[str] = field(default_factory=list)
    improvement_demands: list[str] = field(default_factory=list)
    verdict: str = ""  # REJECT, NEEDS_WORK, ACCEPTABLE, STRONG

    def to_dict(self):
        return asdict(self)


JUDGE_RUBRIC = """
## Adversarial Judge Scoring Rubric

### Thesis Rigor (0-100)
- 90+: Airtight logic, quantified thesis, clear catalysts with timelines
- 70-89: Solid reasoning but some assumptions unvalidated
- 50-69: Surface-level analysis, obvious thesis without differentiated insight
- <50: Weak logic, circular reasoning, or unsupported claims

### Risk Assessment (0-100)
- 90+: Comprehensive risk matrix with probability weighting, tail risks identified
- 70-89: Major risks covered but missing scenario analysis
- 50-69: Generic risks listed without company-specific depth
- <50: Risks ignored or hand-waved away

### Data Quality (0-100)
- 90+: Current financials cited, peer comparisons made, historical context provided
- 70-89: Some data cited but gaps in comparative analysis
- 50-69: Mostly qualitative assertions without quantitative backing
- <50: No data, relying on narrative alone

### Contrarian Thinking (0-100)
- 90+: Actively challenges consensus, considers bear case thoroughly
- 70-89: Acknowledges opposing views but doesn't deeply engage
- 50-69: Consensus-following with token contrarian mention
- <50: Pure momentum/herd thinking

### Actionability (0-100)
- 90+: Clear entry criteria, position sizing rationale, exit conditions
- 70-89: Reasonable framework but missing precision
- 50-69: Vague "buy" recommendation without parameters
- <50: No actionable guidance

## Bias Detection Checklist
- [ ] Recency bias (overweighting recent performance)
- [ ] Survivorship bias (ignoring failed companies in sector)
- [ ] Anchoring bias (fixating on a single metric)
- [ ] Confirmation bias (cherry-picking supporting data)
- [ ] Authority bias (blindly following analyst consensus)
- [ ] Narrative fallacy (compelling story without data)
- [ ] Hindsight bias (rationalizing past moves as predictable)

## Verdict Criteria
- REJECT: Score <50, fundamental flaws, must restart analysis
- NEEDS_WORK: Score 50-69, specific improvements required before acceptance
- ACCEPTABLE: Score 70-84, solid work with minor gaps
- STRONG: Score 85+, institutional-quality research
"""

JUDGE_PROMPT_TEMPLATE = """You are the ADVERSARIAL INVESTMENT JUDGE — the harshest critic on Wall Street.

Your job is to DESTROY weak investment theses. You are NOT here to be nice. You are here to protect capital from lazy thinking.

## Your Persona
- You've seen thousands of bad investment pitches. You're tired of hand-wavy "great company" nonsense.
- You demand EVIDENCE. Not stories. Not narratives. NUMBERS.
- You look for what's MISSING more than what's present.
- You assume every thesis has a fatal flaw until proven otherwise.
- You hate: buzzwords, consensus thinking, ignoring risks, vague catalysts, and "long-term" as a substitute for analysis.

## Scoring Rubric
{rubric}

## Sector Report to Judge
{sector_report}

## Your Task
1. Read every single pick and thesis with extreme skepticism
2. For EACH top pick, ask: "What would make this thesis completely wrong?"
3. Check for ALL biases in the checklist
4. Score each dimension honestly — most research deserves 40-60, not 80+
5. List SPECIFIC improvements needed, not generic advice
6. Your questions should be ones the analyst CANNOT easily answer — that's where the real work needs to happen

Be BRUTAL. Be SPECIFIC. Be USEFUL.

Output your critique as a structured JSON matching the JudgeCritique schema.
"""


def build_judge_prompt(sector_report: dict) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(
        rubric=JUDGE_RUBRIC,
        sector_report=json.dumps(sector_report, indent=2),
    )
