import json
import os
import re
from typing import Any, Dict, List

import streamlit as st


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Multi-Agent AI Interview Panel",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
        .main {
            padding-top: 1rem;
        }

        .app-title {
            font-size: 2.6rem;
            font-weight: 750;
            margin-bottom: 0.2rem;
        }

        .app-subtitle {
            color: #777;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .section-card {
            padding: 1.25rem;
            border-radius: 14px;
            border: 1px solid rgba(128,128,128,0.25);
            margin-bottom: 1rem;
        }

        .agent-card {
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid rgba(128,128,128,0.2);
            margin-bottom: 0.8rem;
        }

        .decision {
            font-size: 2rem;
            font-weight: 750;
            text-transform: uppercase;
        }

        .evidence {
            padding: 0.75rem;
            border-left: 4px solid #888;
            background: rgba(128,128,128,0.08);
            margin: 0.5rem 0;
            border-radius: 5px;
        }

        .change-box {
            padding: 1rem;
            border-radius: 10px;
            border: 2px solid rgba(255,165,0,0.45);
            background: rgba(255,165,0,0.08);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def clean_json(text: str) -> str:
    """Remove accidental markdown code fences."""
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

    return text.strip()


def call_llm(
    client,
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Single isolated LLM call.

    Each agent receives its own call and therefore cannot see
    another agent's conclusion unless that conclusion is
    explicitly passed during the debate stage.
    """

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("LLM returned an empty response.")

    return json.loads(clean_json(content))


# ============================================================
# PROFILE BUILDER
# ============================================================

def build_candidate_profile(
    client,
    resume_text: str,
    transcript_text: str,
    job_description: str,
    model: str,
) -> Dict[str, Any]:

    system_prompt = """
You are the Candidate Profile Builder.

Your job is to extract facts only.

Do NOT evaluate whether the candidate should be hired.

Do NOT invent information.

Every extracted claim must identify its source:
resume, transcript, or both.

If information is missing, explicitly say it is missing.

Return ONLY valid JSON.
"""

    user_prompt = f"""
Create a structured candidate profile.

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

INTERVIEW TRANSCRIPT:
{transcript_text}

Return:

{{
  "candidate_summary": "",
  "skills": [
    {{
      "skill": "",
      "evidence": "",
      "source": "resume|transcript|both"
    }}
  ],
  "experience": [
    {{
      "claim": "",
      "evidence": "",
      "source": "resume|transcript|both"
    }}
  ],
  "education": [],
  "candidate_claims": [
    {{
      "claim": "",
      "evidence": "",
      "source": "resume|transcript|both"
    }}
  ],
  "missing_information": [],
  "job_requirements": [
    {{
      "requirement": "",
      "evidence_status": "supported|partially_supported|not_found",
      "evidence": ""
    }}
  ]
}}
"""

    return call_llm(
        client,
        system_prompt,
        user_prompt,
        model,
        temperature=0.1,
    )


# ============================================================
# AGENT DEFINITIONS
# ============================================================

AGENTS = {
    "Technical Agent": {
        "focus": """
You are the Technical Agent.

Focus ONLY on:
- technical skills
- engineering depth
- architecture
- implementation
- debugging
- relevant tools
- technical claims
- technical requirements from the job description

Do not primarily judge personality or culture.

Be skeptical about shallow keyword matching.

A listed skill alone is NOT proof of deep expertise.
""",
    },

    "HR / Culture Agent": {
        "focus": """
You are the HR / Culture Agent.

Focus ONLY on:
- communication
- teamwork
- leadership
- ownership
- collaboration
- professionalism
- honesty
- ability to explain work
- behavioral interview evidence

Do not make technical depth your primary criterion.

Look carefully for statements that reveal responsibility,
team behavior, communication quality, or honesty.
""",
    },

    "Hiring Manager Agent": {
        "focus": """
You are the Hiring Manager Agent.

Focus on:
- overall role fit
- business value
- relevant experience
- ability to perform the actual job
- strengths that justify hiring
- risks that could make hiring unsafe
- whether the candidate should move forward

You may consider all evidence, but prioritize job requirements
and demonstrated impact.
""",
    },

    "Skeptic Agent": {
        "focus": """
You are the Skeptic Agent.

Your job is to actively search for:
- contradictions
- unsupported claims
- exaggerated expertise
- inconsistencies between resume and transcript
- vague answers
- missing evidence
- suspiciously broad skill claims
- red flags

Do NOT reject someone merely because information is absent.

Distinguish:
1. contradiction
2. unsupported claim
3. genuinely concerning evidence
4. simply missing information
""",
    },
}


# ============================================================
# INDEPENDENT AGENT
# ============================================================

def run_independent_agent(
    client,
    agent_name: str,
    profile: Dict[str, Any],
    job_description: str,
    model: str,
) -> Dict[str, Any]:

    agent = AGENTS[agent_name]

    system_prompt = f"""
{agent["focus"]}

IMPORTANT RULE:

You are working independently.

You MUST NOT assume what another agent thinks.

You have NOT seen any other agent's opinion.

Your evaluation must be based only on:
- the job description
- the candidate profile
- the supplied evidence

Every major conclusion MUST contain evidence.

If evidence is insufficient, say so.

Return ONLY valid JSON.
"""

    user_prompt = f"""
Evaluate the candidate independently.

JOB DESCRIPTION:
{job_description}

CANDIDATE PROFILE:
{json.dumps(profile, indent=2)}

Return:

{{
  "agent": "{agent_name}",
  "decision": "hire|maybe|reject|insufficient_evidence",
  "score": 0,
  "confidence": 0.0,

  "opinion": "",

  "strengths": [
    {{
      "point": "",
      "evidence": {{
        "source": "resume|transcript",
        "quote_or_fact": ""
      }}
    }}
  ],

  "concerns": [
    {{
      "point": "",
      "evidence": {{
        "source": "resume|transcript",
        "quote_or_fact": ""
      }}
    }}
  ],

  "key_evidence": [
    {{
      "source": "resume|transcript",
      "quote_or_fact": "",
      "why_it_matters": ""
    }}
  ],

  "missing_information": [],

  "confidence_reason": ""
}}

SCORING RULE:

The score is NOT arbitrary.

It must reflect the evidence available for your specific role.

Do not fabricate evidence.
"""

    return call_llm(
        client,
        system_prompt,
        user_prompt,
        model,
        temperature=0.25,
    )


# ============================================================
# DEBATE
# ============================================================

def run_debate(
    client,
    profile: Dict[str, Any],
    independent_opinions: Dict[str, Dict[str, Any]],
    job_description: str,
    model: str,
) -> Dict[str, Any]:

    system_prompt = """
You are the Moderator of a multi-agent hiring debate.

The independent opinions have already been produced.

Now agents are allowed to see the other agents' arguments.

Your job is to conduct a REAL debate.

Requirements:

1. At least two agents must directly respond to another agent.
2. Agents must cite the specific argument they are responding to.
3. An agent may agree, disagree, or partially agree.
4. At least one agent should reconsider its position when the evidence
   genuinely warrants it.
5. Do not manufacture disagreement if the evidence clearly agrees.
6. Do not force an opinion change if no evidence justifies it.
7. Every disagreement must trace back to evidence.
8. Missing information must remain missing information.

Return ONLY valid JSON.
"""

    opinions_text = json.dumps(
        independent_opinions,
        indent=2,
    )

    user_prompt = f"""
JOB DESCRIPTION:
{job_description}

CANDIDATE PROFILE:
{json.dumps(profile, indent=2)}

INDEPENDENT AGENT OPINIONS:
{opinions_text}

Conduct a structured debate.

Return:

{{
  "debate_rounds": [
    {{
      "round": 1,
      "speaker": "",
      "responding_to": "",
      "response": "",
      "position": "agree|disagree|partially_agree",
      "evidence": {{
        "source": "resume|transcript",
        "quote_or_fact": ""
      }}
    }}
  ],

  "opinion_changes": [
    {{
      "agent": "",
      "before": "",
      "after": "",
      "changed": true,
      "reason": "",
      "triggering_agent": "",
      "evidence": {{
        "source": "resume|transcript",
        "quote_or_fact": ""
      }}
    }}
  ],

  "consensus_points": [],

  "unresolved_disagreements": [
    {{
      "topic": "",
      "agents": [],
      "why_unresolved": "",
      "evidence": []
    }}
  ]
}}

IMPORTANT:

An opinion change must be genuine.

If no change is justified by the evidence, return changed=false
for that agent rather than inventing one.
"""

    return call_llm(
        client,
        system_prompt,
        user_prompt,
        model,
        temperature=0.35,
    )


# ============================================================
# FINAL DECISION JUDGE
# ============================================================

def final_decision(
    client,
    profile: Dict[str, Any],
    independent_opinions: Dict[str, Dict[str, Any]],
    debate: Dict[str, Any],
    job_description: str,
    model: str,
) -> Dict[str, Any]:

    system_prompt = """
You are the Final Hiring Decision Judge.

You are NOT allowed to calculate the final decision by averaging
agent scores.

Instead, reason over:

1. Job requirements
2. Strength of evidence
3. Evidence quality
4. Agent expertise
5. Contradictions
6. Debate arguments
7. Opinion changes
8. Missing information
9. Unresolved disagreements

A strong piece of direct interview evidence may outweigh several
weak keyword-based resume signals.

A contradiction or serious unsupported claim may materially reduce
confidence.

Missing evidence is NOT automatically negative evidence.

The final recommendation must be:
hire, maybe, or reject.

Every important conclusion must be traceable to evidence.

Return ONLY valid JSON.
"""

    user_prompt = f"""
Make the final hiring decision.

JOB DESCRIPTION:
{job_description}

CANDIDATE PROFILE:
{json.dumps(profile, indent=2)}

INDEPENDENT OPINIONS:
{json.dumps(independent_opinions, indent=2)}

DEBATE:
{json.dumps(debate, indent=2)}

Return:

{{
  "recommendation": "hire|maybe|reject",
  "confidence": 0.0,

  "executive_summary": "",

  "decision_reasoning": [
    {{
      "factor": "",
      "impact": "positive|negative|neutral|uncertain",
      "reason": "",
      "evidence": {{
        "source": "resume|transcript",
        "quote_or_fact": ""
      }}
    }}
  ],

  "strengths": [
    {{
      "point": "",
      "evidence": ""
    }}
  ],

  "concerns": [
    {{
      "point": "",
      "evidence": ""
    }}
  ],

  "decisive_evidence": [
    {{
      "source": "resume|transcript",
      "quote_or_fact": "",
      "impact": "supports_hire|supports_rejection|supports_caution"
    }}
  ],

  "unresolved_disagreements": [],

  "information_gaps": [],

  "agent_changes_that_mattered": [
    {{
      "agent": "",
      "change": "",
      "impact_on_final_decision": ""
    }}
  ],

  "why_not_the_alternative": "",

  "audit_trail": [
    "profile -> independent opinions -> debate -> final decision"
  ]
}}

CRITICAL:

Do NOT average the four scores.

Explain why the evidence was weighted the way it was.
"""

    return call_llm(
        client,
        system_prompt,
        user_prompt,
        model,
        temperature=0.15,
    )


# ============================================================
# COMPLETE CANDIDATE PIPELINE
# ============================================================

def evaluate_candidate(
    client,
    resume_text: str,
    transcript_text: str,
    job_description: str,
    model: str,
) -> Dict[str, Any]:

    # --------------------------------------------------------
    # STEP 1: Candidate Profile
    # --------------------------------------------------------

    profile = build_candidate_profile(
        client,
        resume_text,
        transcript_text,
        job_description,
        model,
    )

    # --------------------------------------------------------
    # STEP 2: Independent Agents
    # --------------------------------------------------------

    independent_opinions = {}

    for agent_name in AGENTS:

        independent_opinions[agent_name] = (
            run_independent_agent(
                client,
                agent_name,
                profile,
                job_description,
                model,
            )
        )

    # --------------------------------------------------------
    # STEP 3: Debate
    # --------------------------------------------------------

    debate = run_debate(
        client,
        profile,
        independent_opinions,
        job_description,
        model,
    )

    # --------------------------------------------------------
    # STEP 4: Final Judge
    # --------------------------------------------------------

    final = final_decision(
        client,
        profile,
        independent_opinions,
        debate,
        job_description,
        model,
    )

    return {
        "profile": profile,
        "independent_opinions": independent_opinions,
        "debate": debate,
        "final_decision": final,
        "metadata": {
            "pipeline": [
                "candidate_profile_builder",
                "technical_agent",
                "hr_culture_agent",
                "hiring_manager_agent",
                "skeptic_agent",
                "debate_moderator",
                "final_decision_judge",
            ],
            "agent_count": 4,
            "final_decision_is_average": False,
        },
    }


# ============================================================
# PDF/TEXT READING
# ============================================================

def read_uploaded_file(uploaded_file) -> str:

    if uploaded_file is None:
        return ""

    try:

        raw = uploaded_file.read()

        file_name = uploaded_file.name.lower()

        # PDF
        if file_name.endswith(".pdf"):

            try:
                import pypdf

                reader = pypdf.PdfReader(
                    uploaded_file
                )

                pages = []

                for page in reader.pages:
                    text = page.extract_text()

                    if text:
                        pages.append(text)

                return "\n".join(pages).strip()

            except Exception:

                return ""

        # TXT / MD
        return raw.decode(
            "utf-8",
            errors="ignore",
        ).strip()

    except Exception:
        return ""


# ============================================================
# LOCAL FALLBACK
# ============================================================

def local_fallback(
    resume: str,
    transcript: str,
    job_description: str,
) -> Dict[str, Any]:

    text = (
        resume
        + "\n"
        + transcript
        + "\n"
        + job_description
    ).lower()

    positive = [
        "built",
        "developed",
        "designed",
        "implemented",
        "deployed",
        "led",
        "production",
        "improved",
        "reduced",
        "increased",
    ]

    concerns = [
        "not sure",
        "unclear",
        "cannot",
        "could not",
        "lack",
        "missing",
        "don't know",
    ]

    positive_count = sum(
        text.count(x)
        for x in positive
    )

    concern_count = sum(
        text.count(x)
        for x in concerns
    )

    score = 55 + min(
        positive_count * 2,
        25,
    )

    score -= min(
        concern_count * 4,
        20,
    )

    score = max(
        0,
        min(
            100,
            score,
        ),
    )

    if score >= 75:
        recommendation = "hire"

    elif score >= 55:
        recommendation = "maybe"

    else:
        recommendation = "reject"

    return {
        "profile": {
            "candidate_summary":
                "Local fallback profile. "
                "LLM profile builder unavailable.",
            "skills": [],
            "experience": [],
            "education": [],
            "candidate_claims": [],
            "missing_information": [
                "Full structured profile unavailable."
            ],
            "job_requirements": [],
        },

        "independent_opinions": {},

        "debate": {
            "debate_rounds": [],
            "opinion_changes": [],
            "consensus_points": [],
            "unresolved_disagreements": [
                "Real multi-agent debate unavailable."
            ],
        },

        "final_decision": {
            "recommendation": recommendation,
            "confidence": 0.5,
            "executive_summary":
                "Fallback evaluation only. "
                "This is not a genuine multi-agent decision.",
            "decision_reasoning": [],
            "strengths": [],
            "concerns": [
                "OpenAI multi-agent pipeline unavailable."
            ],
            "decisive_evidence": [],
            "unresolved_disagreements": [],
            "information_gaps": [],
            "agent_changes_that_mattered": [],
            "why_not_the_alternative": "",
            "audit_trail": [
                "local fallback"
            ],
        },

        "metadata": {
            "pipeline": [
                "local_fallback"
            ],
            "agent_count": 0,
            "final_decision_is_average": False,
        },
    }


# ============================================================
# DISPLAY HELPERS
# ============================================================

def show_evidence(evidence: Dict[str, Any]):

    if not isinstance(evidence, dict):
        st.write(evidence)
        return

    source = evidence.get(
        "source",
        "unknown",
    )

    quote = evidence.get(
        "quote_or_fact",
        evidence.get(
            "evidence",
            "",
        ),
    )

    if quote:

        st.markdown(
            f"""
            <div class="evidence">
            <b>{source.upper()}</b><br>
            "{quote}"
            </div>
            """,
            unsafe_allow_html=True,
        )


def show_profile(profile: Dict[str, Any]):

    st.subheader("📋 Candidate Profile")

    st.write(
        profile.get(
            "candidate_summary",
            "",
        )
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Skills",
            "Experience",
            "Claims",
            "Missing Information",
        ]
    )

    with tab1:

        for item in profile.get(
            "skills",
            [],
        ):

            st.markdown(
                f"**{item.get('skill', '')}**"
            )

            st.caption(
                item.get(
                    "evidence",
                    "",
                )
            )

    with tab2:

        for item in profile.get(
            "experience",
            [],
        ):

            st.markdown(
                f"**{item.get('claim', '')}**"
            )

            st.caption(
                item.get(
                    "evidence",
                    "",
                )
            )

    with tab3:

        for item in profile.get(
            "candidate_claims",
            [],
        ):

            st.markdown(
                f"**{item.get('claim', '')}**"
            )

            st.caption(
                item.get(
                    "evidence",
                    "",
                )
            )

    with tab4:

        gaps = profile.get(
            "missing_information",
            [],
        )

        if gaps:

            for gap in gaps:
                st.warning(str(gap))

        else:
            st.success(
                "No major information gaps identified."
            )


def show_independent_agents(
    opinions: Dict[str, Dict[str, Any]]
):

    st.subheader(
        "🤖 Independent Agent Opinions"
    )

    st.caption(
        "These opinions were generated in separate LLM calls "
        "before any agent saw another agent's conclusion."
    )

    for name, opinion in opinions.items():

        with st.container(border=True):

            st.markdown(
                f"### {name}"
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Decision",
                    str(
                        opinion.get(
                            "decision",
                            "unknown",
                        )
                    ).upper(),
                )

            with col2:
                st.metric(
                    "Score",
                    f"{float(opinion.get('score', 0)):.1f}/100",
                )

            with col3:
                st.metric(
                    "Confidence",
                    f"{float(opinion.get('confidence', 0)):.0%}",
                )

            st.write(
                opinion.get(
                    "opinion",
                    "",
                )
            )

            strengths = opinion.get(
                "strengths",
                [],
            )

            if strengths:

                st.markdown(
                    "**Strengths**"
                )

                for item in strengths:

                    st.write(
                        f"• {item.get('point', '')}"
                    )

                    show_evidence(
                        item.get(
                            "evidence",
                            {},
                        )
                    )

            concerns = opinion.get(
                "concerns",
                [],
            )

            if concerns:

                st.markdown(
                    "**Concerns**"
                )

                for item in concerns:

                    st.write(
                        f"• {item.get('point', '')}"
                    )

                    show_evidence(
                        item.get(
                            "evidence",
                            {},
                        )
                    )

            missing = opinion.get(
                "missing_information",
                [],
            )

            if missing:

                st.info(
                    "Missing information: "
                    + "; ".join(
                        map(str, missing)
                    )
                )


def show_debate(debate: Dict[str, Any]):

    st.subheader(
        "🗣️ Agent Debate"
    )

    st.caption(
        "Agents can now see and directly respond to "
        "other agents' arguments."
    )

    rounds = debate.get(
        "debate_rounds",
        [],
    )

    if not rounds:

        st.info(
            "No debate rounds were returned."
        )

    for item in rounds:

        speaker = item.get(
            "speaker",
            "Agent",
        )

        responding_to = item.get(
            "responding_to",
            "",
        )

        position = item.get(
            "position",
            "",
        )

        st.markdown(
            f"### {speaker}"
        )

        if responding_to:

            st.caption(
                f"Responding to: {responding_to}"
            )

        st.write(
            item.get(
                "response",
                "",
            )
        )

        st.info(
            f"Position: {position}"
        )

        show_evidence(
            item.get(
                "evidence",
                {},
            )
        )

    st.markdown(
        "### 🔄 Opinion Changes"
    )

    changes = debate.get(
        "opinion_changes",
        [],
    )

    if not changes:

        st.info(
            "No opinion changes were reported."
        )

    for change in changes:

        if change.get(
            "changed",
            False,
        ):

            st.markdown(
                f"""
                <div class="change-box">
                <b>{change.get('agent', '')}</b><br><br>
                Before: {change.get('before', '')}<br>
                After: {change.get('after', '')}<br><br>
                Reason: {change.get('reason', '')}<br>
                Triggered by: {change.get('triggering_agent', '')}
                </div>
                """,
                unsafe_allow_html=True,
            )

            show_evidence(
                change.get(
                    "evidence",
                    {},
                )
            )

    unresolved = debate.get(
        "unresolved_disagreements",
        [],
    )

    if unresolved:

        st.markdown(
            "### ⚖️ Unresolved Disagreements"
        )

        for item in unresolved:

            st.warning(
                f"**{item.get('topic', '')}** — "
                f"{item.get('why_unresolved', '')}"
            )


def show_final_decision(final: Dict[str, Any]):

    st.markdown("---")

    st.subheader(
        "🏁 Final Decision"
    )

    recommendation = final.get(
        "recommendation",
        "maybe",
    )

    confidence = float(
        final.get(
            "confidence",
            0,
        )
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Recommendation",
            str(
                recommendation
            ).upper(),
        )

    with col2:

        st.metric(
            "Confidence",
            f"{confidence:.0%}",
        )

    st.markdown(
        "### Executive Summary"
    )

    st.write(
        final.get(
            "executive_summary",
            "",
        )
    )

    st.markdown(
        "### Decision Reasoning"
    )

    for item in final.get(
        "decision_reasoning",
        [],
    ):

        impact = item.get(
            "impact",
            "neutral",
        )

        st.markdown(
            f"**{item.get('factor', '')}** "
            f"({impact})"
        )

        st.write(
            item.get(
                "reason",
                "",
            )
        )

        show_evidence(
            item.get(
                "evidence",
                {},
            )
        )

    left, right = st.columns(2)

    with left:

        st.markdown(
            "### ✅ Strengths"
        )

        for item in final.get(
            "strengths",
            [],
        ):

            st.success(
                item.get(
                    "point",
                    str(item),
                )
            )

            if isinstance(item, dict):
                st.caption(
                    item.get(
                        "evidence",
                        "",
                    )
                )

    with right:

        st.markdown(
            "### ⚠️ Concerns"
        )

        for item in final.get(
            "concerns",
            [],
        ):

            st.warning(
                item.get(
                    "point",
                    str(item),
                )
            )

            if isinstance(item, dict):
                st.caption(
                    item.get(
                        "evidence",
                        "",
                    )
                )

    st.markdown(
        "### 🎯 Decisive Evidence"
    )

    decisive = final.get(
        "decisive_evidence",
        [],
    )

    if decisive:

        for item in decisive:

            show_evidence(item)

            st.caption(
                item.get(
                    "impact",
                    "",
                )
            )

    else:

        st.info(
            "No decisive evidence was returned."
        )

    gaps = final.get(
        "information_gaps",
        [],
    )

    if gaps:

        st.markdown(
            "### 🔍 Information Gaps"
        )

        for gap in gaps:
            st.info(str(gap))

    st.markdown(
        "### ⚖️ Remaining Disagreement"
    )

    unresolved = final.get(
        "unresolved_disagreements",
        [],
    )

    if unresolved:

        for item in unresolved:
            st.warning(str(item))

    else:

        st.success(
            "No unresolved disagreement was reported."
        )

    st.markdown(
        "### Why Not the Alternative?"
    )

    st.write(
        final.get(
            "why_not_the_alternative",
            "",
        )
    )


# ============================================================
# SESSION STATE
# ============================================================

if "candidate_results" not in st.session_state:
    st.session_state.candidate_results = {}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "🤖 Multi-Agent Interview Panel"
    )

    st.write(
        "Evidence-based AI candidate evaluation."
    )

    st.divider()

    api_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv(
            "OPENAI_API_KEY",
            "",
        ),
        type="password",
        help="Required for the multi-agent LLM pipeline.",
    )

    model = st.text_input(
        "Model",
        value="gpt-4o-mini",
    )

    st.divider()

    st.markdown(
        """
        **Pipeline**

        1. Candidate Profile Builder
        2. Technical Agent
        3. HR / Culture Agent
        4. Hiring Manager Agent
        5. Skeptic Agent
        6. Independent opinions
        7. Debate
        8. Opinion-change tracking
        9. Final Decision Judge
        """
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="app-title">🤖 Multi-Agent AI Interview Panel</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-subtitle">
    Four independent AI personas analyze each candidate,
    debate conflicting evidence, and produce an auditable
    final hiring recommendation.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# JOB DESCRIPTION
# ============================================================

st.header(
    "1️⃣ Job Description"
)

job_description_file = st.file_uploader(
    "Upload Job Description PDF/TXT",
    type=[
        "pdf",
        "txt",
        "md",
    ],
    key="job_description_file",
)

job_description = st.text_area(
    "Or paste the Job Description",
    height=250,
    key="job_description_text",
)

if job_description_file:

    extracted = read_uploaded_file(
        job_description_file
    )

    if extracted:

        job_description = extracted

        st.success(
            f"Loaded: {job_description_file.name}"
        )


# ============================================================
# CANDIDATE INPUTS
# ============================================================

st.header(
    "2️⃣ Candidate Documents"
)

candidate_tabs = st.tabs(
    [
        "Candidate A",
        "Candidate B",
    ]
)


candidate_inputs = {}


for index, candidate_name in enumerate(
    ["A", "B"]
):

    with candidate_tabs[index]:

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "Resume"
            )

            resume_file = st.file_uploader(
                f"Upload Resume {candidate_name}",
                type=[
                    "pdf",
                    "txt",
                    "md",
                ],
                key=f"resume_{candidate_name}",
            )

            resume_text = st.text_area(
                f"Paste Resume {candidate_name}",
                height=300,
                key=f"resume_text_{candidate_name}",
            )

            if resume_file:

                extracted = read_uploaded_file(
                    resume_file
                )

                if extracted:

                    resume_text = extracted

                    st.success(
                        f"Loaded: {resume_file.name}"
                    )

        with col2:

            st.subheader(
                "Interview Transcript"
            )

            transcript_file = st.file_uploader(
                f"Upload Transcript {candidate_name}",
                type=[
                    "pdf",
                    "txt",
                    "md",
                ],
                key=f"transcript_{candidate_name}",
            )

            transcript_text = st.text_area(
                f"Paste Transcript {candidate_name}",
                height=300,
                key=f"transcript_text_{candidate_name}",
            )

            if transcript_file:

                extracted = read_uploaded_file(
                    transcript_file
                )

                if extracted:

                    transcript_text = extracted

                    st.success(
                        f"Loaded: {transcript_file.name}"
                    )

        candidate_inputs[candidate_name] = {
            "resume": resume_text,
            "transcript": transcript_text,
        }


# ============================================================
# EVALUATE
# ============================================================

st.markdown("---")

evaluate = st.button(
    "🚀 Run Multi-Agent Evaluation",
    type="primary",
    use_container_width=True,
)


if evaluate:

    if not job_description.strip():

        st.error(
            "Please provide the Job Description."
        )

    else:

        valid_candidates = []

        for candidate_name, data in candidate_inputs.items():

            if (
                data["resume"].strip()
                and data["transcript"].strip()
            ):

                valid_candidates.append(
                    candidate_name
                )

        if not valid_candidates:

            st.error(
                "Please provide at least one complete candidate "
                "resume and interview transcript."
            )

        elif not api_key:

            st.warning(
                "No OpenAI API key was provided. "
                "Running the local fallback evaluator. "
                "The fallback is NOT the full multi-agent pipeline."
            )

            for candidate_name in valid_candidates:

                data = candidate_inputs[
                    candidate_name
                ]

                st.session_state.candidate_results[
                    candidate_name
                ] = local_fallback(
                    data["resume"],
                    data["transcript"],
                    job_description,
                )

        else:

            try:

                from openai import OpenAI

                client = OpenAI(
                    api_key=api_key
                )

                progress = st.progress(
                    0
                )

                status = st.empty()

                for i, candidate_name in enumerate(
                    valid_candidates
                ):

                    status.write(
                        f"Evaluating Candidate {candidate_name}..."
                    )

                    data = candidate_inputs[
                        candidate_name
                    ]

                    result = evaluate_candidate(
                        client,
                        data["resume"],
                        data["transcript"],
                        job_description,
                        model,
                    )

                    st.session_state.candidate_results[
                        candidate_name
                    ] = result

                    progress.progress(
                        (i + 1)
                        / len(valid_candidates)
                    )

                status.success(
                    "Multi-agent evaluation complete."
                )

            except Exception as exc:

                st.error(
                    "Multi-agent evaluation failed."
                )

                st.exception(exc)


# ============================================================
# RESULTS
# ============================================================

if st.session_state.candidate_results:

    st.markdown("---")

    st.header(
        "📊 Evaluation Results"
    )

    result_tabs = st.tabs(
        [
            f"Candidate {name}"
            for name
            in st.session_state.candidate_results
        ]
    )

    for tab, candidate_name in zip(
        result_tabs,
        st.session_state.candidate_results,
    ):

        with tab:

            result = (
                st.session_state
                .candidate_results[
                    candidate_name
                ]
            )

            st.subheader(
                f"Candidate {candidate_name}"
            )

            # ------------------------------------------------
            # Audit pipeline
            # ------------------------------------------------

            metadata = result.get(
                "metadata",
                {},
            )

            pipeline = metadata.get(
                "pipeline",
                [],
            )

            if pipeline:

                st.caption(
                    " → ".join(
                        pipeline
                    )
                )

            # ------------------------------------------------
            # Profile
            # ------------------------------------------------

            show_profile(
                result.get(
                    "profile",
                    {},
                )
            )

            # ------------------------------------------------
            # Independent opinions
            # ------------------------------------------------

            show_independent_agents(
                result.get(
                    "independent_opinions",
                    {},
                )
            )

            # ------------------------------------------------
            # Debate
            # ------------------------------------------------

            show_debate(
                result.get(
                    "debate",
                    {},
                )
            )

            # ------------------------------------------------
            # Final decision
            # ------------------------------------------------

            show_final_decision(
                result.get(
                    "final_decision",
                    {},
                )
            )

            # ------------------------------------------------
            # JSON
            # ------------------------------------------------

            with st.expander(
                "🔎 View Complete Audit JSON"
            ):

                st.json(
                    result
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Multi-Agent AI Interview Panel Simulator • "
    "Evidence → Independent Analysis → Debate → Final Decision"
)
