::: {align="center"}
# Return Clasification Tool

### AI-assisted e-commerce returns analysis

**Built by Syncraft Technologies**

A web-based platform that helps administrators review return requests,
examine product evidence, and use Gemini-powered analysis to support
more consistent, evidence-based decisions.

[🌐 Live Website](https://return-clasification-tool.onrender.com/login) · [🧠
Architecture](#-system-architecture)
:::

------------------------------------------------------------------------

## Overview

The **Return Intelligence Platform** brings return-case information,
product images, AI-generated findings, and administrator review into one
workflow.

The application uses a Flask backend, PostgreSQL for application
records, Supabase Storage for images when configured, and Google's
Gemini API for case reasoning. The AI can interpret submitted case
information and available images, highlight supporting or conflicting
evidence, identify missing information, and recommend a next action.

> **Important:** AI output is decision support, not proof of fraud or an
> automatic final verdict. Administrators should review ambiguous,
> incomplete, or consequential cases.

## ✨ Key Features

-   **Return-case management** --- organize and review submitted return
    requests.
-   **AI-assisted analysis** --- use Gemini to produce structured
    findings from case details and available evidence.
-   **Image-aware review** --- analyze listing and received-product
    images when supported images are successfully supplied to the model.
-   **Evidence-focused output** --- surface supporting evidence,
    contradictions, alternative explanations, missing information, and
    suggested next steps where provided by the workflow.
-   **Human-review safeguards** --- flag cases where evidence is weak,
    insufficient, contradictory, or incomplete.
-   **Administrator dashboard** --- review request status, analysis
    results, and operational summaries.
-   **Background processing** --- run analysis jobs outside the
    immediate request flow where configured.
-   **Persistent storage** --- use PostgreSQL for application data and
    Supabase Storage for images when configured.
-   **Quality reporting** --- reporting functionality for reviewing case
    analysis, depending on enabled dashboard routes and templates.

## 🧰 Technology Stack

  -----------------------------------------------------------------------
  Layer                               Technology
  ----------------------------------- -----------------------------------
  Backend                             Python, Flask

  Database                            PostgreSQL

  ORM / database integration          SQLAlchemy, Flask-SQLAlchemy

  Authentication                      Flask-Login

  AI reasoning                        Google Gemini API

  Image storage                       Supabase Storage, when configured

  AI / image utilities                Python imaging and model libraries,
                                      where enabled

  Frontend                            HTML, CSS, JavaScript, Jinja
                                      templates

  Background analysis                 Python background worker /
                                      executor, where configured
  -----------------------------------------------------------------------

## 🧠 System Architecture

``` mermaid
flowchart TD
    A[Return case submitted] --> B[Flask application: app.py]
    B --> C[(PostgreSQL database)]
    B --> D[Image storage: Supabase or configured storage]
    B --> E[Background analysis job]
    E --> F[ai_analysis.py]
    F --> G[Prepare case details, allowed reasons, evidence and image paths]
    G --> H[gemini_reasoner.py]
    H --> I[Google Gemini API]
    I --> H
    H --> J[Validate and normalize structured response]
    J --> K[Save analysis result to PostgreSQL]
    K --> L[Administrator dashboard and quality report]
    L --> M[Human review and final decision]
```

### Main Components

-   **`app.py`** --- Flask application entry point. Handles routes,
    authentication, database interaction, dashboard pages, and the
    analysis workflow.
-   **`ai_analysis.py`** --- coordinates case analysis. Prepares case
    evidence and valid return reasons, supplies available image paths
    and optional similar cases, and calls the reasoning layer.
-   **`gemini_reasoner.py`** --- integrates with the Gemini API, builds
    the reasoning prompt, requests a structured response, and applies
    response validation and error-handling logic.
-   **`config.py`** --- holds application configuration, including
    database and AI/storage settings. Secrets should be supplied through
    environment variables rather than committed to source control.
-   **Templates and static assets** --- provide the browser interface,
    dashboard, case details, and related styling.

### AI Analysis Workflow

1.  A return case and its available evidence enter the Flask
    application.
2.  The application stores case data and schedules analysis when
    background processing is enabled.
3.  `ai_analysis.py` assembles the case context, permitted return-reason
    categories, relevant evidence, and available image paths.
4.  `gemini_reasoner.py` sends the prompt and supported image inputs to
    the configured Gemini model.
5.  The response is parsed and validated into the structure expected by
    the application.
6.  The application persists the result and displays it to an
    administrator for review.

**Model behavior and limitations:** Gemini is a pretrained model
accessed through an API; this project does not imply that Gemini was
trained or fine-tuned on the application's return database. Supplying
historical cases as context is not model training. Image findings depend
on image quality, successful image loading, and the information actually
provided to the model. Do not treat an AI-generated assessment or
uncalibrated score as a calibrated fraud probability.

## 🌐 Live Website

**Production URL:** [www.return=clasification-tool.com](https://return-clasification-tool.onrender.com/login)


## ⚙️ Local Setup

These are general setup steps. Exact package names, commands, and
environment-variable names should match the current project files.

### 1. Clone the repository

``` bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd <YOUR-REPOSITORY-DIRECTORY>
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

``` powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**

``` bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

If the repository contains `requirements.txt`:

``` bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a local `.env` file if the project uses `python-dotenv`, or
configure the variables in your deployment provider. Use the exact
variable names expected by `config.py` and the AI/storage modules.

Example values below are illustrative; verify names against your current
code:

``` dotenv
SECRET_KEY=replace-with-a-strong-secret
SQLALCHEMY_DATABASE_URI=your-postgresql-connection-string
GEMINI_API_KEY=your-gemini-api-key
SUPABASE_URL=your-supabase-project-url
SUPABASE_KEY=your-supabase-key
SUPABASE_STORAGE_BUCKET=your-configured-bucket-name
```

Never commit real secrets, production credentials, customer data, or
private image URLs.

### 5. Initialize the database if required

Use the database initialization or migration procedure documented by the
project. Do not run destructive schema commands against a production
database.

### 6. Start the application

If `app.py` is the configured entry point:

``` bash
python app.py
```

Open the local address printed by Flask. In production, use the
deployment platform's configured application server rather than Flask's
development server.

## 🧪 Testing and Quality

Before releasing changes, verify that:

-   New return cases can be submitted and persisted.
-   Valid and invalid image uploads are handled safely.
-   AI analysis succeeds when the Gemini service is available.
-   API errors, rate limits, and unreadable images result in clear
    failure states or human-review flags.
-   Background jobs update the correct case and do not leave cases stuck
    indefinitely.
-   Administrator-only routes are protected.
-   Dashboard summaries and quality reports match the database.
-   Database connections recover appropriately after interruptions.

## 🚧 Known Considerations

-   Gemini responses can be incorrect or incomplete and should be
    reviewed.
-   Lighting, camera angle, background, and compression can make image
    comparisons misleading.
-   A photo alone may not establish when damage occurred or prove a
    hidden defect or product functionality.
-   Historical cases influence an analysis only if the application
    retrieves and supplies them as context.
-   A model's self-reported confidence or a placeholder confidence field
    is not a calibrated probability unless validated with a suitable
    evaluation process.

## 🗺️ Future Improvements

Potential next steps include an evaluation dataset of
administrator-verified cases, measured false-positive and false-negative
rates, calibrated confidence estimates, audit trails linking findings to
evidence, robust job retries and monitoring, and expanded analytics.

## 👥 Maintainer

**Tushar Dhamecha**

-   ** Social website:** [www.Linkedin.in](https://www.linkedin.com/in/tushar-dhamecha/)
-   **Application:** Add your deployed application URL in the [Live
    Website](https://return-clasification-tool.onrender.com/admin/returns/9) section.

------------------------------------------------------------------------

::: {align="center"}
**Return Intelligence Platform**\
*Evidence-informed return analysis with human oversight.*
:::
