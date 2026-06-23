from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BenchmarkQuestion:
    question: str
    expected_docs: tuple[str, ...]


BENCHMARK_QUESTIONS = (
    BenchmarkQuestion(
        "Reconcile the claim that US EV demand was slowing in early 2024 with evidence that policy and charging access still supported adoption.",
        ("doc_1.txt", "doc_2.txt", "doc_36.txt", "doc_48.txt"),
    ),
    BenchmarkQuestion(
        "Compare consumer hesitation about EVs with dealer sentiment: which barriers appear demand-side, and which are channel-side?",
        ("doc_21.txt", "doc_23.txt", "doc_35.txt", "doc_40.txt", "doc_44.txt"),
    ),
    BenchmarkQuestion(
        "Explain why charging infrastructure can matter both for consumer adoption and for investor confidence in the EV market.",
        ("doc_27.txt", "doc_48.txt", "doc_70.txt"),
    ),
    BenchmarkQuestion(
        "Across the US, Europe, and China, which EV sales patterns suggest a temporary slowdown versus a structural regional advantage?",
        ("doc_67.txt", "doc_69.txt", "doc_63.txt", "doc_64.txt"),
    ),
    BenchmarkQuestion(
        "Contrast Tesla's competitive pressure with the broader field of Chinese EV entrants and explain what this implies for market structure.",
        ("doc_22.txt", "doc_61.txt", "doc_63.txt", "doc_66.txt"),
    ),
    BenchmarkQuestion(
        "Use company financial results to compare whether Polestar, VinFast, and ZEEKR show growth, stress, or mixed signals.",
        ("doc_12.txt", "doc_13.txt", "doc_14.txt", "doc_15.txt"),
    ),
    BenchmarkQuestion(
        "What evidence supports the argument that EV policy is not just a subsidy story but also a market-shaping industrial strategy?",
        ("doc_1.txt", "doc_6.txt", "doc_63.txt", "doc_64.txt", "doc_66.txt"),
    ),
    BenchmarkQuestion(
        "Identify the strongest evidence against simplistic anti-EV narratives about batteries, reliability, and environmental benefits.",
        ("doc_3.txt", "doc_8.txt", "doc_17.txt", "doc_38.txt"),
    ),
    BenchmarkQuestion(
        "If a policymaker wanted to increase EV adoption, which combination of model availability, incentives, and charging access is best supported by the corpus?",
        ("doc_1.txt", "doc_35.txt", "doc_40.txt", "doc_48.txt", "doc_70.txt"),
    ),
    BenchmarkQuestion(
        "Explain how EV investment and job creation claims relate to concerns that the US auto industry could fall behind.",
        ("doc_6.txt", "doc_38.txt", "doc_63.txt", "doc_64.txt"),
    ),
    BenchmarkQuestion(
        "Distinguish between evidence about actual EV sales volumes and evidence about sentiment toward EVs in the US market.",
        ("doc_2.txt", "doc_21.txt", "doc_23.txt", "doc_30.txt", "doc_36.txt"),
    ),
    BenchmarkQuestion(
        "What cross-document evidence suggests that investor sentiment toward EV companies is not explained by revenue growth alone?",
        ("doc_12.txt", "doc_13.txt", "doc_14.txt", "doc_15.txt", "doc_22.txt"),
    ),
    BenchmarkQuestion(
        "Compare government policy in China with US state and local policy: where do they overlap, and where do they differ in effect?",
        ("doc_1.txt", "doc_63.txt", "doc_64.txt", "doc_66.txt"),
    ),
    BenchmarkQuestion(
        "How do public-health and environmental arguments for EVs interact with adoption barriers reported by consumers?",
        ("doc_3.txt", "doc_17.txt", "doc_35.txt", "doc_40.txt", "doc_44.txt"),
    ),
    BenchmarkQuestion(
        "Which documents together best explain why EV adoption can rise in some metropolitan areas while national sentiment remains cautious?",
        ("doc_1.txt", "doc_21.txt", "doc_35.txt", "doc_44.txt", "doc_48.txt"),
    ),
    BenchmarkQuestion(
        "Assess whether the EV market's main challenge is supply, demand, infrastructure, or finance, using evidence from at least three document groups.",
        ("doc_2.txt", "doc_15.txt", "doc_35.txt", "doc_48.txt", "doc_70.txt"),
    ),
    BenchmarkQuestion(
        "What evidence connects EV model availability requirements with broader sales outcomes and regional adoption differences?",
        ("doc_1.txt", "doc_30.txt", "doc_67.txt", "doc_69.txt"),
    ),
    BenchmarkQuestion(
        "Explain why Chinese EV battery dominance matters for US automakers, beyond simple competition between car brands.",
        ("doc_38.txt", "doc_61.txt", "doc_63.txt", "doc_64.txt", "doc_66.txt"),
    ),
    BenchmarkQuestion(
        "Find the strongest contradiction between optimistic EV adoption signals and skeptical market signals, then resolve it using document evidence.",
        ("doc_1.txt", "doc_2.txt", "doc_21.txt", "doc_36.txt", "doc_44.txt"),
    ),
    BenchmarkQuestion(
        "Build a concise investment thesis for and against the EV sector using sales trends, policy support, infrastructure, and company results.",
        ("doc_6.txt", "doc_12.txt", "doc_13.txt", "doc_15.txt", "doc_27.txt", "doc_67.txt"),
    ),
)
