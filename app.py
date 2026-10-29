
from pathlib import Path
import re

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="SkillMap AI",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# 2. DARK BLUE DESIGN SYSTEM
# ==================================================

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
    }

    .stApp {
        background: #080F1F;
        color: #E2E8F0;
    }

    [data-testid="stHeader"] {
        background: rgba(8, 15, 31, 0.95);
    }

    [data-testid="stSidebar"] {
        background: #0D1729;
        border-right: 1px solid #20304B;
    }

    [data-testid="stSidebar"] * {
        color: #D5E1F2;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #F1F5F9 !important;
        letter-spacing: -0.4px;
    }

    p, label, li {
        color: #C0CDDF;
    }

    [data-testid="stCaptionContainer"] {
        color: #91A4BF;
    }

    .hero {
        background: linear-gradient(
            125deg,
            #102344 0%,
            #101C35 55%,
            #102B4C 100%
        );
        border: 1px solid #24466F;
        border-radius: 20px;
        padding: 32px;
        margin-bottom: 24px;
    }

    .eyebrow {
        color: #60A5FA;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero-title {
        font-size: clamp(30px, 4vw, 44px);
        font-weight: 750;
        line-height: 1.15;
        color: #F8FAFC;
        margin-bottom: 12px;
    }

    .hero-description {
        font-size: 16px;
        line-height: 1.7;
        color: #B5C6DD;
        max-width: 700px;
    }

    .section-label {
        color: #60A5FA;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .panel {
        background: #0D1729;
        border: 1px solid #20304B;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 16px;
    }

    .career-card {
        background: #0D1729;
        border: 1px solid #253854;
        border-radius: 16px;
        padding: 24px;
        margin-top: 14px;
        margin-bottom: 20px;
    }

    .score-label {
        color: #60A5FA;
        font-size: 13px;
        font-weight: 650;
    }

    .muted {
        color: #91A4BF;
        font-size: 13px;
        line-height: 1.6;
    }

    .tag {
        display: inline-block;
        background: #142947;
        color: #93C5FD;
        border: 1px solid #24466F;
        border-radius: 7px;
        padding: 5px 9px;
        margin: 3px 4px 3px 0;
        font-size: 12px;
    }

    [data-testid="stMetric"] {
        background: #0D1729;
        border: 1px solid #20304B;
        border-radius: 14px;
        padding: 18px;
    }

    [data-testid="stMetricLabel"] {
        color: #91A4BF !important;
    }

    [data-testid="stMetricValue"] {
        color: #60A5FA !important;
    }

    [data-testid="stProgressBar"] > div {
        background: #20304B;
        border-radius: 20px;
    }

    [data-testid="stProgressBar"] > div > div {
        background: #3B82F6;
        border-radius: 20px;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        background: #2563EB;
        color: #FFFFFF;
        border: 1px solid #3B82F6;
        border-radius: 10px;
        min-height: 44px;
        font-weight: 650;
        transition: 0.2s ease;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background: #1D4ED8;
        border-color: #60A5FA;
        color: #FFFFFF;
    }

    .stTextInput input,
    .stTextArea textarea,
    [data-baseweb="select"] > div {
        background: #101C30 !important;
        color: #E2E8F0 !important;
        border-color: #2B3E5C !important;
        border-radius: 9px !important;
    }

    div[data-baseweb="select"] * {
        color: #E2E8F0 !important;
    }

    [data-testid="stExpander"] {
        background: #0D1729;
        border: 1px solid #20304B;
        border-radius: 12px;
    }

    hr {
        border-color: #20304B;
    }

    a {
        color: #60A5FA !important;
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    .footer {
        border-top: 1px solid #20304B;
        margin-top: 40px;
        padding-top: 20px;
        color: #7185A2;
        font-size: 12px;
        line-height: 1.8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# 3. LOAD CAREER DATA
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "careers.csv"

REQUIRED_COLUMNS = [
    "career_title",
    "description",
    "required_skills",
    "beginner_projects",
    "learning_resources",
]


@st.cache_data
def load_careers():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Cannot find data/careers.csv. "
            "Keep the data folder beside app.py."
        )

    data = pd.read_csv(DATA_PATH)

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            "Missing CSV columns: " + ", ".join(missing)
        )

    data = data.dropna(subset=["career_title"]).copy()

    for col in REQUIRED_COLUMNS:
        data[col] = data[col].fillna("").astype(str)

    data = data.drop_duplicates(
        subset=["career_title"]
    )

    if data.empty:
        raise ValueError("No career profiles found in the CSV.")

    return data.reset_index(drop=True)


try:
    careers = load_careers()
except Exception as error:
    st.error(f"Unable to load the dataset: {error}")
    st.stop()


# ==================================================
# 4. SKILL PROCESSING
# ==================================================

def split_items(value):
    if not value:
        return []

    return [
        item.strip()
        for item in re.split(r"[;,]", str(value))
        if item.strip()
    ]


def normalize(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9+#. ]", " ", text)
    text = re.sub(r"\s+", " ", text)

    aliases = {
        "python basics": "python",
        "basic python": "python",
        "ai basics": "artificial intelligence",
        "ai fundamentals": "artificial intelligence",
        "ml basics": "machine learning",
        "ml fundamentals": "machine learning",
        "ui ux": "user experience design",
        "ui/ux": "user experience design",
    }

    return aliases.get(text, text)


def compare_skills(user_skills, required_skills):
    matched = []
    missing = []

    normalized_user = [
        normalize(skill) for skill in user_skills
    ]

    for skill in required_skills:
        target = normalize(skill)

        found = any(
            user_skill == target
            or (
                len(target) >= 4
                and target in user_skill
            )
            or (
                len(user_skill) >= 4
                and user_skill in target
            )
            for user_skill in normalized_user
        )

        if found:
            matched.append(skill)
        else:
            missing.append(skill)

    return matched, missing


# ==================================================
# 5. LEARNING RESOURCES
# ==================================================

RESOURCE_LINKS = {
    "Google Machine Learning Crash Course":
        "https://developers.google.com/machine-learning/crash-course",
    "Google ML Crash Course":
        "https://developers.google.com/machine-learning/crash-course",
    "Kaggle Learn Python":
        "https://www.kaggle.com/learn/python",
    "Kaggle Learn Pandas":
        "https://www.kaggle.com/learn/pandas",
    "scikit-learn documentation":
        "https://scikit-learn.org/stable/user_guide.html",
    "Python official tutorial":
        "https://docs.python.org/3/tutorial/",
    "MDN Web Docs":
        "https://developer.mozilla.org/",
    "Microsoft Learn":
        "https://learn.microsoft.com/training/",
    "Google UX Design resources":
        "https://grow.google/certificates/ux-design/",
    "Figma learning resources":
        "https://help.figma.com/hc/en-us/categories/360002051613",
    "Cisco Networking Academy":
        "https://www.netacad.com/",
    "Kaggle":
        "https://www.kaggle.com/learn",
    "GitHub":
        "https://github.com/",
}


def show_resources(resource_text):
    resources = split_items(resource_text)

    if not resources:
        st.write("No learning resources listed yet.")
        return

    for resource in resources:
        url = RESOURCE_LINKS.get(resource)

        if url:
            st.markdown(f"- [{resource}]({url})")
        else:
            st.markdown(f"- {resource}")


# ==================================================
# 6. TF-IDF + COSINE SIMILARITY ENGINE
# ==================================================

def recommend_careers(profile_text, career_data, top_n=3):
    career_texts = []

    for _, career in career_data.iterrows():
        text = " ".join([
            career["career_title"],
            career["description"],
            career["required_skills"],
        ])
        career_texts.append(text)

    documents = [profile_text] + career_texts

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )

    try:
        matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return career_data.head(0).copy(), 0

    profile_vector = matrix[0:1]
    career_vectors = matrix[1:]

    scores = cosine_similarity(
        profile_vector,
        career_vectors,
    ).flatten()

    results = career_data.copy()
    results["similarity_score"] = scores

    results = results.sort_values(
        "similarity_score",
        ascending=False,
    )

    feature_count = len(vectorizer.get_feature_names_out())

    return results.head(top_n), feature_count


# ==================================================
# 7. SIDEBAR
# ==================================================

with st.sidebar:
    st.markdown("## SkillMap AI")
    st.caption("Career exploration workspace")

    st.divider()

    st.markdown("### Workspace")
    st.write("Career discovery")
    st.write("Skill-gap analysis")
    st.write("Learning roadmap")

    st.divider()

    st.markdown("### About this project")
    st.caption(
        "A student-built career recommendation prototype "
        "using Python, Streamlit and natural language processing."
    )

    st.caption(
        f"Dataset currently contains {len(careers)} career profiles."
    )

    if st.button("Clear results", use_container_width=True):
        for key in [
            "recommendations",
            "profile_text",
            "current_skills",
            "career_goal",
            "education",
            "feature_count",
        ]:
            st.session_state.pop(key, None)

        st.rerun()


# ==================================================
# 8. HEADER
# ==================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">CAREER INTELLIGENCE PLATFORM</div>
        <div class="hero-title">Find your direction.<br>Build your next step.</div>
        <div class="hero-description">
            Explore career paths based on your interests and skills.
            Discover skill gaps, practical projects and learning resources
            to help plan your journey.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# 9. AI EXPLANATION
# ==================================================

with st.expander("How the recommendation engine works"):
    st.markdown("#### Text processing with TF-IDF")

    st.write(
        "TF-IDF converts words in your profile and career descriptions "
        "into weighted numerical features. This lets the application "
        "compare text mathematically."
    )

    st.markdown("#### Ranking with cosine similarity")

    st.write(
        "Cosine similarity compares your profile vector with each "
        "career vector. The application ranks career profiles by "
        "their text similarity and displays the top matches."
    )

    st.markdown("#### Separate skill-gap analysis")

    st.write(
        "A separate rule-based comparison checks the skills you enter "
        "against the required skills listed in each career profile."
    )

    st.info(
        "This is a content-based recommendation prototype, not a "
        "trained hiring classifier. Similarity scores are not hiring "
        "probabilities, and the dataset is not an official qualification standard."
    )


# ==================================================
# 10. PROFILE FORM
# ==================================================

st.markdown(
    '<div class="section-label">01 / YOUR PROFILE</div>',
    unsafe_allow_html=True,
)

st.header("Build your career profile")

st.write(
    "Share your interests and current skills to explore relevant career paths."
)

with st.form("profile_form"):
    col1, col2 = st.columns(2)

    with col1:
        education = st.selectbox(
            "Education level",
            [
                "High school student",
                "Undergraduate student",
                "Postgraduate student",
                "Graduate",
                "Working professional",
            ],
        )

        career_options = ["Not sure yet"] + sorted(
            careers["career_title"].unique().tolist()
        )

        career_goal = st.selectbox(
            "Career of interest",
            career_options,
        )

    with col2:
        interests = st.text_area(
            "Interests",
            placeholder=(
                "For example: AI, business, problem-solving, "
                "building useful products"
            ),
            height=100,
        )

        current_skills = st.text_area(
            "Current skills",
            placeholder=(
                "For example: Python basics, communication, "
                "AI fundamentals"
            ),
            height=100,
            help="Separate skills with commas.",
        )

    submitted = st.form_submit_button(
        "Generate career matches",
        use_container_width=True,
    )


# ==================================================
# 11. PROCESS THE PROFILE
# ==================================================

if submitted:
    if not interests.strip() and not current_skills.strip():
        st.warning(
            "Enter at least some interests or current skills "
            "before generating recommendations."
        )
    else:
        profile_parts = [
            interests.strip(),
            current_skills.strip(),
            education,
        ]

        if career_goal != "Not sure yet":
            profile_parts.append(career_goal)

        profile_text = " ".join(profile_parts)

        recommendations, feature_count = recommend_careers(
            profile_text,
            careers,
            top_n=3,
        )

        if recommendations.empty:
            st.error(
                "The profile could not be processed. "
                "Try adding more information."
            )
        else:
            st.session_state["recommendations"] = recommendations
            st.session_state["profile_text"] = profile_text
            st.session_state["current_skills"] = current_skills
            st.session_state["career_goal"] = career_goal
            st.session_state["education"] = education
            st.session_state["feature_count"] = feature_count


# ==================================================
# 12. RECOMMENDATION RESULTS
# ==================================================

if "recommendations" in st.session_state:
    recommendations = st.session_state["recommendations"]
    user_skills = split_items(
        st.session_state["current_skills"]
    )

    st.divider()

    st.markdown(
        '<div class="section-label">02 / YOUR RESULTS</div>',
        unsafe_allow_html=True,
    )

    st.header("Your career matches")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric("Profiles compared", len(careers))

    with metric2:
        st.metric("Top matches", len(recommendations))

    with metric3:
        st.metric(
            "Text features",
            st.session_state.get("feature_count", 0),
        )

    st.caption(
        "Text features are the distinct terms and term combinations "
        "used by TF-IDF for this comparison."
    )

    st.markdown("### Recommended career paths")

    for rank, (_, career) in enumerate(
        recommendations.iterrows(),
        start=1,
    ):
        title = career["career_title"]
        score = float(career["similarity_score"])
        required_skills = split_items(career["required_skills"])

        matched, missing = compare_skills(
            user_skills,
            required_skills,
        )

        with st.container(border=True):
            title_col, score_col = st.columns([3, 1])

            with title_col:
                st.markdown(f"### {rank:02d}  {title}")

            with score_col:
                st.markdown(
                    f'<div class="score-label">'
                    f'TEXT SIMILARITY<br>{score * 100:.1f}%'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.write(career["description"])

            st.progress(
                max(0.0, min(1.0, score))
            )

            st.caption(
                "Relative text similarity only. This percentage "
                "does not represent hiring probability, skill mastery "
                "or job eligibility."
            )

            with st.expander("Explore skills, projects and resources"):
                left, right = st.columns(2)

                with left:
                    st.markdown("#### Skills already listed")

                    if matched:
                        for skill in matched:
                            st.markdown(
                                f'<span class="tag">{skill}</span>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.write(
                            "No direct skill matches were detected. "
                            "This does not mean you have no relevant skills."
                        )

                with right:
                    st.markdown("#### Skills to explore")

                    if missing:
                        for skill in missing:
                            st.markdown(
                                f'<span class="tag">{skill}</span>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.write(
                            "All listed skills matched the skills "
                            "in this career profile."
                        )

                st.divider()

                st.markdown("#### Beginner project ideas")
                projects = split_items(career["beginner_projects"])

                if projects:
                    for project in projects:
                        st.write(f"- {project}")
                else:
                    st.write("No project ideas listed yet.")

                st.markdown("#### Learning resources")
                show_resources(career["learning_resources"])


    # ==================================================
    # 13. LEARNING ROADMAP
    # ==================================================

    st.divider()

    st.markdown(
        '<div class="section-label">03 / YOUR ACTION PLAN</div>',
        unsafe_allow_html=True,
    )

    st.header("Your learning roadmap")

    selected_goal = st.session_state.get(
        "career_goal",
        "Not sure yet",
    )

    roadmap_career = None

    if selected_goal != "Not sure yet":
        selected_rows = careers[
            careers["career_title"] == selected_goal
        ]

        if not selected_rows.empty:
            roadmap_career = selected_rows.iloc[0]

    if roadmap_career is None:
        roadmap_career = recommendations.iloc[0]

    st.write(
        f"Suggested starting point: **{roadmap_career['career_title']}**"
    )

    roadmap_skills = split_items(
        roadmap_career["required_skills"]
    )

    st.write(
        "Use this checklist to organise your next learning steps."
    )

    if roadmap_skills:
        for index, skill in enumerate(roadmap_skills, start=1):
            st.checkbox(
                f"{index:02d}  Learn or practise {skill}",
                key=f"roadmap_{roadmap_career['career_title']}_{index}",
            )
    else:
        st.write("No roadmap skills are listed for this career yet.")

    st.caption(
        "Checklist selections are temporary and are not saved "
        "between app sessions. This roadmap is a guide, not a guarantee "
        "of employment."
    )


# ==================================================
# 14. FOOTER
# ==================================================

st.markdown(
    """
    <div class="footer">
        <strong>SkillMap AI</strong><br>
        AI/ML career exploration prototype<br>
        Built with Python, Streamlit, Pandas and scikit-learn.
    </div>
    """,
    unsafe_allow_html=True,
)
