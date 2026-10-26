# Return Intelligence Platform

AI-assisted return analysis and staff verification platform for e-commerce operations.

[🌐 Live Demo](https://return-clasification-tool.onrender.com/login) · [🧠 Architecture](#-system-architecture)

A web-based workflow for reviewing return requests, validating product evidence, and using Gemini-powered analysis to support more consistent, evidence-based decisions.

> AI-generated findings are decision support tools, not final proof of fraud or an automatic verdict. Human review remains essential for ambiguous, incomplete, or consequential cases.

## Overview

The Return Intelligence Platform brings together return-case information, product images, AI-generated findings, and administrator review in a single workflow.

The application uses:

- Flask for the web application and API layer
- PostgreSQL for application data
- Supabase Storage for image persistence when configured
- Google Gemini API for structured evidence-based analysis

This helps teams assess return cases faster while preserving human oversight and auditability.

## Key Features

- Return-case intake and review workflow
- AI-assisted reasoning using case details and available evidence
- Product listing and received-item image review
- Structured findings with supporting evidence, contradictions, missing information, and recommended next actions
- Human-review safeguards for weak or incomplete evidence
- Admin dashboard for case triage, review, and quality reporting
- Background processing for asynchronous AI analysis
- Persistent storage for case records and image assets

## Technology Stack

| Layer | Technology |
| --- | --- |
| Backend | Python, Flask |
| Database | PostgreSQL, SQLAlchemy, Flask-SQLAlchemy |
| Authentication | Flask-Login |
| AI reasoning | Google Gemini API |
| Image storage | Supabase Storage (optional) |
| Frontend | HTML, CSS, JavaScript, Jinja templates |
| Background jobs | Python executor / worker pattern |

## System Architecture

```mermaid
flowchart TD
    A[Return case submitted] --> B[Flask app: app.py]
    B --> C[(PostgreSQL database)]
    B --> D[Image storage: Supabase or configured storage]
    B --> E[Background analysis job]
    E --> F[ai_analysis.py]
    F --> G[Prepare case context, valid reasons, and evidence]
    G --> H[gemini_reasoner.py]
    H --> I[Google Gemini API]
    I --> H
    H --> J[Validate and normalize structured response]
    J --> K[Save analysis result to PostgreSQL]
    K --> L[Administrator dashboard and quality report]
    L --> M[Human review and final decision]
```

### Main Components

- `app.py` — Flask application entry point, routes, authentication flow, dashboard pages, and case review logic
- `ai_analysis.py` — assembles evidence, available images, and prompt context for AI reasoning
- `gemini_reasoner.py` — integrates with Gemini, builds prompts, and validates structured AI output
- `config.py` — environment-driven configuration for the app, database, and storage
- `models/` — data models for users, return requests, images, and analysis results
- `templates/` and `static/` — frontend presentation and styling

### AI Analysis Workflow

1. A return case and supporting evidence are submitted through the app.
2. The system stores the case and queues analysis when enabled.
3. `ai_analysis.py` prepares the context, allowed reasons, and images for the model.
4. `gemini_reasoner.py` sends the request to Gemini and normalizes the response.
5. The application saves the predicted outcome and surfaces it in the admin workflow.
6. Administrators review the findings before finalizing a decision.

## Repository Structure

```text
.
├── app.py
├── ai_analysis.py
├── config.py
├── extensions.py
├── gemini_reasoner.py
├── models/
├── static/
├── templates/
├── requirements.txt
├── Procfile
├── README.md
├── .gitignore
├── test_gemini_prompt.py
├── upload_images_to_supabase.py
├── migrate_sqlite_to_supabase.py
└── ai_analysis_backup/
```

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/ImTusharr/return-intelligence-platform.git
cd return-intelligence-platform
```

### 2. Create and activate a virtual environment

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root and set the required values. Use the exact names expected by the application code.

```env
SECRET_KEY=replace-with-a-strong-secret-key
DATABASE_URL=postgresql://user:password@host:5432/database
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
SUPABASE_STORAGE_BUCKET=return-images
```

Notes:

- `DATABASE_URL` defaults to SQLite when not set in local development.
- Never commit real secrets, customer data, or private image URLs to source control.

### 5. Initialize the database

The project includes a Flask CLI command for creating tables:

```bash
python app.py
```

or, if using the Flask application factory pattern from a shell environment:

```bash
flask --app app init-db
```

### 6. Start the application

```bash
python app.py
```

Then open the local Flask URL shown in the terminal.

## Testing and Quality Checks

Before releasing changes, confirm that:

- Return cases can be submitted and saved correctly
- Image uploads validate as expected
- AI analysis succeeds when Gemini is available
- API failures and rate limits are handled gracefully
- Background task updates do not leave cases stuck in a processing state
- Admin-only routes are protected
- Dashboard and quality-report metrics reflect current database state

## Important Considerations

- Gemini responses can be incomplete or incorrect and should always be reviewed by a human
- Image quality, lighting, angle, and compression can affect AI interpretation
- A photo alone may not prove fault timing, hidden defects, or product functionality
- AI output should support staff decisions rather than replace them

## Future Improvements

Potential next steps include:

- Verified evaluation data for model accuracy over time
- Better confidence calibration and reporting
- Stronger audit trails linking findings to evidence
- Improved retry and monitoring for background jobs
- Expanded analytics and admin reporting

## Maintainer

**Tushar Dhamecha**

- LinkedIn: https://www.linkedin.com/in/tushar-dhamecha/
- Live demo: https://return-clasification-tool.onrender.com/login

---

Return Intelligence Platform

Evidence-informed return analysis with human oversight.
