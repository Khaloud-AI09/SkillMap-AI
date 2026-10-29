
# SkillMap AI

### Find your direction. Build your next step.

SkillMap AI is a student-built career exploration and skill-gap recommendation application. It helps users explore career paths based on their interests and current skills, discover skills to develop, and find beginner-friendly projects and learning resources.

The application uses Python, Streamlit, Pandas, and scikit-learn to provide content-based career recommendations through text processing and similarity scoring.

## Overview

Choosing a career can be difficult when you are unsure which roles fit your interests or what skills to learn next.

SkillMap AI aims to make career exploration more structured by allowing users to enter their interests, education level, career preference, and current skills. It then compares their profile with a collection of career descriptions and displays relevant career suggestions.

## Features

- **Career recommendations:** Ranks career profiles based on text similarity to the user's profile.
- **Skill-gap analysis:** Compares user-entered skills with the skills listed for recommended careers.
- **Beginner project ideas:** Suggests practical projects to help users begin exploring a career.
- **Learning resources:** Provides learning links and resources where available.
- **Learning roadmap:** Displays a checklist of skills to explore for a selected career.
- **Interactive interface:** A dark navy and blue-themed web interface built with Streamlit.
- **Multiple career profiles:** Includes a curated dataset of 25 career profiles.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| Streamlit | Interactive web application |
| Pandas | Dataset loading and processing |
| scikit-learn | TF-IDF vectorisation and cosine similarity |
| CSV | Career profile dataset |

## How the Recommendation Engine Works

### 1. User profile

The user enters their interests, education level, career preference, and current skills.

### 2. Text processing with TF-IDF

The application combines each career's title, description, and listed required skills into a text profile. TF-IDF converts the user's profile and career descriptions into numerical text features.

### 3. Similarity scoring

Cosine similarity compares the user's profile vector with the career profile vectors. The application ranks the career profiles by their similarity scores and displays the top three matches.

### 4. Skill-gap comparison

A separate rule-based text comparison identifies skills that match the user's entered skills and skills that may be worth exploring.

### 5. Learning suggestions

The application displays beginner project ideas and learning resources from the career dataset, along with a checklist of skills for the user's learning roadmap.

## Dataset

The application uses a curated CSV dataset located at `data/careers.csv`.

Each career profile contains:

- `career_title`
- `description`
- `required_skills`
- `beginner_projects`
- `learning_resources`

The dataset includes a range of careers in artificial intelligence, data, software development, design, cybersecurity, cloud computing, business, and related fields.

The dataset is a prototype resource, not an official occupational standard or verified statement of every employer's requirements.

## Getting Started

### Prerequisites

- Python 3.10 or a compatible Python version
- pip
- Git (optional, for cloning the repository)

### 1. Clone the repository

Replace `YOUR_GITHUB_USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/SkillMap-AI.git
cd SkillMap-AI
```

Alternatively, download the repository as a ZIP file and extract it.

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the application using the virtual environment's Python executable directly.

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL in your terminal. Open that URL in your browser.

## Project Structure

```text
SkillMap-AI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── careers.csv
├── src/
└── screenshots/
```

Additional files may be present depending on the project setup.

## Limitations

- Recommendations are based on text similarity, not a trained hiring classifier.
- Similarity scores do not represent hiring probability, skill mastery, or job eligibility.
- The skill-gap comparison uses straightforward text matching and may not recognise every synonym.
- Recommendations depend on the content and coverage of the curated dataset.
- Learning resources and career requirements should be checked for current relevance.

SkillMap AI is intended for educational career exploration and planning. It does not replace professional career counselling.

## Future Improvements

Potential future improvements include:

- Expanding and validating the career dataset.
- Improving skill matching with synonym handling.
- Adding user accounts and persistent learning progress.
- Evaluating recommendation quality with structured test profiles.
- Improving accessibility and adding additional language support.

## Author

Developed as a student AI/ML project to explore content-based recommendations and interactive applications using Python and scikit-learn.

## License

No license has been specified yet. Unless a license is added, the repository should not be assumed to permit unrestricted reuse or redistribution.
