

# Adaptive Personal Assistant (Django)

![Chatbot interface screenshot](assets/K12_Chatbot.png) 
*Chatbot interface (tutor answer + related images).*

## Project Overview

This repository contains a Django-based adaptive tutoring web application.

The app supports a guided learning flow:

1. A user registers or logs in
2. A first-time user completes a one-time preference setup
3. The user creates a learning goal
4. The system classifies that goal and generates a knowledge base
5. The user submits a self-assessment
6. The user enters the tutoring chat
7. After each answer, the user completes an understanding check before moving on
8. Each chat answer is enriched with relevant images from Wikimedia Commons

Main responsibilities of the system:

- User registration, login, Google OAuth login, and Django admin access
- MySQL-backed storage through Django models and migrations
- Learning-goal classification and knowledge-base generation
- Self-assessment generation and evaluation
- Adaptive chat with understanding checks and answer-style weighting
- **Image search** — Wikimedia Commons API (primary) with local ChromaDB/OpenCLIP fallback
- **Dual LLM support** — runs with Ollama (free, local) or OpenAI (paid), controlled by a single env var

Tech stack:

- Django 4.2
- MySQL
- Django ORM + Django migrations
- LLM: Ollama (free, default) or OpenAI API
- Image search: Wikimedia Commons API + ChromaDB / OpenCLIP
- HTML templates + static CSS

Notes:

MAIN REPO - https://github.com/johnleddoMETY/chatbot/tree/Django-Version
This repo only contains the changes we did to the main repository.

- Database schema is managed by Django models in `adaptive-virtual-assistant-beta/ava_apps/main/models.py`
- Migrations live in `adaptive-virtual-assistant-beta/ava_apps/main/migrations/`
- Standard Django admin is the supported admin interface at `/django-admin/`
- All LLM calls are routed through a central client at `ava_apps/core/services/llm_client.py`

## Project Structure

```text
AdaptivePersonalAssistant/
├── README.md
└── adaptive-virtual-assistant-beta/
    ├── ava_django/
    │   ├── settings.py              # Django settings
    │   ├── urls.py                  # Root URL config
    │   └── __init__.py              # PyMySQL bootstrap
    ├── ava_apps/
    │   ├── accounts/
    │   │   ├── forms.py             # Register/login/setup-preferences forms
    │   │   ├── urls.py              # Auth routes
    │   │   ├── auth_backends.py     # Custom auth backend against users table
    │   │   └── services/            # Register/login/preferences/google OAuth logic
    │   ├── main/
    │   │   ├── views.py             # Main entry views for all page routes
    │   │   ├── urls.py              # Main app routes
    │   │   ├── models.py            # All business tables
    │   │   └── admin.py             # Django admin registrations
    │   ├── learning_goal/
    │   │   ├── flow_service.py      # Learning-goal step orchestration
    │   │   ├── review_service.py    # Goal text validation
    │   │   ├── categorization_service.py
    │   │   ├── kb_generation_service.py
    │   │   ├── goal_repository_service.py
    │   │   └── services/            # Learning-goal data + KB generation services
    │   ├── self_assessment/
    │   │   ├── flow_service.py      # Self-assessment step orchestration
    │   │   ├── goal_context_service.py
    │   │   ├── generation_service.py
    │   │   ├── evaluation_service.py
    │   │   ├── persistence_service.py
    │   │   └── services/            # LLM evaluation and priority logic
    │   ├── chat/
    │   │   ├── general_chat/        # Main tutoring chat flow
    │   │   │   ├── answer_flow_service.py      # Orchestrates Q→answer→images
    │   │   │   └── services/
    │   │   │       ├── answer_generation_service.py  # LLM answer generation
    │   │   │       ├── image_search_service.py       # ChromaDB local image search (fallback)
    │   │   │       └── wikimedia_image_service.py    # Wikimedia Commons API search (primary)
    │   │   ├── check_understanding/ # Understanding-check flow
    │   │   └── shared/              # Shared conversation-memory helpers
    │   ├── core/
    │   │   └── services/
    │   │       ├── database_service.py      # Shared DB service layer
    │   │       ├── knowledge_hub_service.py # Shared KB support assembly
    │   │       └── llm_client.py            # Central LLM client (Ollama / OpenAI)
    │   └── admin_portal/
    │       └── urls.py              # Placeholder custom admin portal routes
    ├── templates_django/
    │   ├── base.html
    │   ├── index.html
    │   ├── self_assessment.html
    │   ├── accounts/
    │   ├── learning_goal/
    │   └── chat/
    ├── static/
    │   └── css/                     # Per-page CSS (includes image gallery styles)
    ├── scripts/
    │   └── start_django.sh          # Convenience startup script
    ├── tests/                       # Test suite
    ├── utils/
    │   └── nlp_utils.py             # NLP helper functions
    ├── requirements-django.txt
    ├── .env.example
    └── manage.py
```

## Setup and Run

### 1. Prerequisites

Install:

- Python `3.10+`
- MySQL
- [Ollama](https://ollama.ai) (for free local LLM) **or** an OpenAI API key with credits

### 2. Clone and enter the Django app directory

```bash
git clone <your-repo-url>
cd AdaptivePersonalAssistant/adaptive-virtual-assistant-beta
```

### 3. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

All remaining commands below assume the virtual environment is activated.

### 4. Install dependencies

```bash
pip install -r requirements-django.txt
```

### 5. Configure environment variables

Copy `.env.example` to `.env`, then fill in your real values.

```bash
cp .env.example .env
```

Minimum required values:

```env
DB_HOST=127.0.0.1
DB_PORT=3306
DB_NAME=Django-Adaptive-Assistant
DB_USER=adaptive_user
DB_PASSWORD=your_db_password

SECRET_KEY=replace_with_random_string

# LLM provider: "ollama" (free, default) or "openai" (paid)
LLM_PROVIDER=ollama

# Only required when LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-...
```

Optional values used by the app:

- `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI` — Google OAuth
- `OLLAMA_BASE_URL` — Ollama server URL (default: `http://localhost:11434/v1`)
- `OLLAMA_MODEL` — Ollama model for heavy tasks (default: `llama3.2`)
- `OLLAMA_FAST_MODEL` — Ollama model for light tasks (default: same as `OLLAMA_MODEL`)
- `OPENAI_MODEL` — OpenAI model for heavy tasks (default: `gpt-4.1-2025-04-14`)
- `OPENAI_FAST_MODEL` — OpenAI model for light tasks (default: `gpt-4o-mini`)

### 6. Create the MySQL database

The database itself must exist before migration. Migrations will create all tables inside it.

Example:

```sql
CREATE DATABASE `Django-Adaptive-Assistant`
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;
```

Your MySQL user must have normal DDL permissions on that database.

### 7. Run migrations

```bash
python manage.py migrate --noinput
```

This is the standard schema workflow. No extra manual table creation is required after the database exists.

### 8. Set up the LLM

**Option A — Ollama (free, recommended for development):**

```bash
brew install ollama          # macOS (or see https://ollama.ai for other OS)
ollama serve                 # start the server (keep running in a separate terminal)
ollama pull llama3.2         # download the model (~2 GB, one-time)
```

**Option B — OpenAI (paid):**

Set `LLM_PROVIDER=openai` and a valid `OPENAI_API_KEY` in your `.env` file.

### 9. Create a Django admin user

```bash
python manage.py createsuperuser
```

### 10. Start the server

```bash
python manage.py runserver 127.0.0.1:8002
```

Or use the helper script:

```bash
bash scripts/start_django.sh
```

### 11. Open the app

- Main app: `http://127.0.0.1:8002/`
- Django admin: `http://127.0.0.1:8002/django-admin/`

## Database and Schema Workflow

This project uses a standard Django schema workflow:

1. Edit models in `adaptive-virtual-assistant-beta/ava_apps/main/models.py`
2. Generate migrations
3. Apply migrations

Commands:

```bash
python manage.py makemigrations
python manage.py migrate
```

That is the only schema workflow the team should use going forward.

## Main Pages, Steps, and Python Functions

This section maps the main user-facing flow to the Python entrypoints and service functions behind each step.

### 1. Landing Page (`/`)

Template:

- `adaptive-virtual-assistant-beta/templates_django/index.html`

Main function:

- `adaptive-virtual-assistant-beta/ava_apps/main/views.py -> index()`
  - Renders the landing page

Related function:

- `adaptive-virtual-assistant-beta/ava_apps/main/views.py -> about()`
  - Currently renders the same landing template for `/about`

### 2. Register (`/auth/register`)

Template:

- `adaptive-virtual-assistant-beta/templates_django/accounts/register.html`

Main functions:

- `adaptive-virtual-assistant-beta/ava_apps/main/views.py -> register_view()`
  - Handles registration form submit, creates the app user, then auto-logs in the user
- `adaptive-virtual-assistant-beta/ava_apps/accounts/services/registration_service.py -> register_new_user()`
  - Checks uniqueness and creates a new user profile
- `adaptive-virtual-assistant-beta/ava_apps/accounts/services/registration_service.py -> _build_new_user_payload()`
  - Builds the default user payload stored in MySQL
- `adaptive-virtual-assistant-beta/ava_apps/core/services/database_service.py -> create_user()`
  - Persists the user record and related list data

### 3. Login (`/auth/login`)

Template:

- `adaptive-virtual-assistant-beta/templates_django/accounts/login.html`

Main functions:

- `adaptive-virtual-assistant-beta/ava_apps/main/views.py -> login_view()`
  - Handles the login form and redirects based on setup state
- `adaptive-virtual-assistant-beta/ava_apps/accounts/services/login_service.py -> authenticate_login_user()`
  - Decides whether the user goes to setup-preferences or directly to learning-goal
- `adaptive-virtual-assistant-beta/ava_apps/accounts/auth_backends.py -> DatabaseServiceBackend.authenticate()`
  - Validates credentials against the `users` table
- `adaptive-virtual-assistant-beta/ava_apps/accounts/auth_backends.py -> DatabaseServiceBackend._get_or_sync_django_user()`
  - Creates or syncs the Django `auth_user` record for session login

### 4. Google Login (`/auth/login/google`, `/auth/google/callback`)

Main functions:

- `adaptive-virtual-assistant-beta/ava_apps/main/views.py -> google_login()`
  - Starts the Google OAuth flow
- `adaptive-virtual-assistant-beta/ava_apps/main/views.py -> google_callback()`
  - Handles the OAuth callback, creates or finds the local user, and logs them in
- `adaptive-virtual-assistant-beta/ava_apps/accounts/services/google_oauth_service.py -> build_google_authorize_url()`
  - Builds the Google authorization URL and state token
- `adaptive-virtual-assistant-beta/ava_apps/accounts/services/google_oauth_service.py -> exchange_code_for_userinfo()`
  - Exchanges the code for Google profile information
- `adaptive-virtual-assistant-beta/ava_apps/accounts/services/google_oauth_service.py -> ensure_profile_for_google_user()`
  - Creates a local app profile if the email does not exist yet
- `adaptive-virtual-assistant-beta/ava_apps/accounts/services/google_oauth_service.py -> sync_django_user()`
  - Syncs the Django auth user for session login

### 5. First-Time Preferences (`/auth/setup-preferences`)

Template:

- `adaptive-virtual-assistant-beta/templates_django/accounts/setup_preferences.html`

Main functions:

- `adaptive-virtual-assistant-beta/ava_apps/main/views.py -> setup_preferences()`
  - Handles the one-time setup form for `age` and `academic_level`
- `adaptive-virtual-assistant-beta/ava_apps/accounts/services/preferences_service.py -> build_preferences_initial_data()`
  - Pre-fills the form from the user profile
- `adaptive-virtual-assistant-beta/ava_apps/accounts/services/preferences_service.py -> build_preferences_update_data()`
  - Converts cleaned form data into DB update payload
- `adaptive-virtual-assistant-beta/ava_apps/accounts/services/preferences_service.py -> save_user_preferences()`
  - Saves the setup data and marks `preferences_completed=True`

### 6. Learning Goal Page (`/learning-goal`)

Template:

- `adaptive-virtual-assistant-beta/templates_django/learning_goal/page.html`
- `adaptive-virtual-assistant-beta/templates_django/learning_goal/sidebar.html`

Main functions:

- `adaptive-virtual-assistant-beta/ava_apps/main/views.py -> learning_goal()`
  - Shows the page and handles creation of a new goal
- `adaptive-virtual-assistant-beta/ava_apps/learning_goal/flow_service.py -> create_learning_goal_from_preference()`
  - Orchestrates validation, classification, KB generation, and DB creation
- `adaptive-virtual-assistant-beta/ava_apps/learning_goal/review_service.py -> review_preference_text()`
  - Validates and cleans the raw goal text
- `adaptive-virtual-assistant-beta/ava_apps/learning_goal/categorization_service.py -> categorize_learning_goal()`
  - Classifies the goal into `domain` and `branch`
- `adaptive-virtual-assistant-beta/ava_apps/learning_goal/kb_generation_service.py -> generate_learning_goal_kb()`
  - Triggers knowledge-base creation for the classified goal
- `adaptive-virtual-assistant-beta/ava_apps/learning_goal/goal_repository_service.py -> create_learning_goal_record()`
  - Persists the learning goal record
- `adaptive-virtual-assistant-beta/ava_apps/learning_goal/flow_service.py -> list_user_learning_goals()`
  - Loads the sidebar learning-goal list

### 7. Open Existing Learning Goal (`/learning-goal/<id>/open`)

Main functions:

- `adaptive-virtual-assistant-beta/ava_apps/main/views.py -> open_learning_goal()`
  - Opens an existing goal and redirects to the correct next step
- `adaptive-virtual-assistant-beta/ava_apps/learning_goal/flow_service.py -> resolve_open_learning_goal()`
  - Decides whether the user should go to self-assessment or chat

### 8. Self-Assessment (`/self-assessment?learning_goal_id=...`)

Template:

- `adaptive-virtual-assistant-beta/templates_django/self_assessment.html`
- `adaptive-virtual-assistant-beta/templates_django/learning_goal/sidebar.html`

Main functions:

- `adaptive-virtual-assistant-beta/ava_apps/main/views.py -> self_assessment()`
  - Displays the self-assessment page and handles submit
- `adaptive-virtual-assistant-beta/ava_apps/self_assessment/flow_service.py -> resolve_request_goal_context()`
  - Resolves the current goal, domain, branch, and preference text
- `adaptive-virtual-assistant-beta/ava_apps/self_assessment/flow_service.py -> build_template_context()`
  - Builds the page context, example text, KB snippets, and sidebar data
- `adaptive-virtual-assistant-beta/ava_apps/self_assessment/flow_service.py -> submit_self_assessment()`
  - Builds the assessment payload, runs evaluation, and persists the result
- `adaptive-virtual-assistant-beta/ava_apps/self_assessment/generation_service.py -> build_or_load_example_text()`
  - Loads or generates the example self-assessment text
- `adaptive-virtual-assistant-beta/ava_apps/self_assessment/generation_service.py -> load_knowledge_base_content()`
  - Loads KB content for display on the page
- `adaptive-virtual-assistant-beta/ava_apps/self_assessment/evaluation_service.py -> build_assessment_payload()`
  - Builds the structured input for the evaluation pipeline
- `adaptive-virtual-assistant-beta/ava_apps/self_assessment/evaluation_service.py -> run_self_assessment_evaluation()`
  - Executes the self-assessment evaluation service
- `adaptive-virtual-assistant-beta/ava_apps/self_assessment/persistence_service.py -> save_evaluation_record()`
  - Writes the evaluation result to `self_assessments`
- `adaptive-virtual-assistant-beta/ava_apps/self_assessment/persistence_service.py -> mark_goal_completed()`
  - Marks the learning goal as ready for chat

### 9. Chat Page (`/chat?learning_goal_id=...`)

Template:

- `adaptive-virtual-assistant-beta/templates_django/chat/page.html`
- `adaptive-virtual-assistant-beta/templates_django/chat/input_form.html`
- `adaptive-virtual-assistant-beta/templates_django/chat/load_more.html`
- `adaptive-virtual-assistant-beta/templates_django/chat/knowledge_check_panel.html`
- `adaptive-virtual-assistant-beta/templates_django/learning_goal/sidebar.html`

Main functions:

- `adaptive-virtual-assistant-beta/ava_apps/main/views.py -> chat()`
  - Displays the chat page for the active learning goal
- `adaptive-virtual-assistant-beta/ava_apps/chat/general_chat/page_service.py -> resolve_chat_access()`
  - Verifies the goal exists and that self-assessment is completed
- `adaptive-virtual-assistant-beta/ava_apps/chat/general_chat/page_service.py -> build_chat_page_context()`
  - Builds chat history, sidebar goals, and understanding-check state
- `adaptive-virtual-assistant-beta/ava_apps/main/views.py -> chat_history()`
  - Returns paginated history for the chat page
- `adaptive-virtual-assistant-beta/ava_apps/chat/general_chat/page_service.py -> load_chat_history_page()`
  - Loads paginated user history records from the database

### 10. Get Tutor Answer (`/get_answer`)

Main functions:

- `adaptive-virtual-assistant-beta/ava_apps/main/views.py -> get_answer()`
  - Accepts a question and returns a tutor answer as JSON
- `adaptive-virtual-assistant-beta/ava_apps/chat/general_chat/answer_flow_service.py -> answer_question_flow()`
  - Main orchestration for answer generation, image search, history save, and pending understanding state
- `adaptive-virtual-assistant-beta/ava_apps/chat/general_chat/services/answer_generation_service.py -> get_ai_answer()`
  - Calls the LLM (Ollama or OpenAI via `llm_client`) and generates the adaptive tutor answer
- `adaptive-virtual-assistant-beta/ava_apps/chat/general_chat/services/wikimedia_image_service.py -> search_wikimedia_images()`
  - Searches Wikimedia Commons for relevant educational images (no API key needed)
- `adaptive-virtual-assistant-beta/ava_apps/chat/general_chat/services/image_search_service.py -> search_images()`
  - Fallback: searches the local ChromaDB image database using OpenCLIP embeddings
- `adaptive-virtual-assistant-beta/ava_apps/core/services/database_service.py -> add_user_history()`
  - Saves the question/answer pair (including image gallery HTML) in `user_histories`
- `adaptive-virtual-assistant-beta/ava_apps/learning_goal/services/learning_goal_service.py -> set_goal_pending_remediation()`
  - Stores the pending understanding-check state for the current goal

### 11. Understanding Check (`/generate_key_points`, `/check_understanding`, `/skip_understanding`)

Main functions:

- `adaptive-virtual-assistant-beta/ava_apps/main/views.py -> generate_key_points_async()`
  - Starts background key-point extraction for the latest answer
- `adaptive-virtual-assistant-beta/ava_apps/chat/check_understanding/flow_service.py -> generate_key_points_async_flow()`
  - Extracts and stores the answer key points
- `adaptive-virtual-assistant-beta/ava_apps/main/views.py -> check_understanding()`
  - Accepts the learner summary and returns the evaluation result
- `adaptive-virtual-assistant-beta/ava_apps/chat/check_understanding/flow_service.py -> check_understanding_flow()`
  - Scores the summary, updates weights, updates pending state, and stores the turn
- `adaptive-virtual-assistant-beta/ava_apps/chat/check_understanding/evaluation_service.py -> evaluate_understanding_with_reasons()`
  - Uses the LLM to compare key points against the learner summary
- `adaptive-virtual-assistant-beta/ava_apps/chat/shared/conversation_memory_service.py -> record_understanding_turn()`
  - Appends the understanding-check turn to the remote conversation
- `adaptive-virtual-assistant-beta/ava_apps/main/views.py -> skip_understanding()`
  - Clears the current understanding-check state
- `adaptive-virtual-assistant-beta/ava_apps/chat/check_understanding/flow_service.py -> skip_understanding_flow()`
  - Removes the pending remediation state from the learning goal

### 12. Django Admin (`/django-admin/`)

Main file:

- `adaptive-virtual-assistant-beta/ava_apps/main/admin.py`
  - Registers all core models with Django admin so tables can be managed through the standard admin UI

Registered core models:

- `UserProfile`
- `UserHobby`
- `UserGoal`
- `UserStrength`
- `UserWeakness`
- `UserBadge`
- `AnswerTypeWeight`
- `LearningGoal`
- `UserHistory`
- `SelfAssessment`
- `UserConversationState`
- `KnowledgeBaseEntry`

Important note:

- `adaptive-virtual-assistant-beta/ava_apps/admin_portal/` is a placeholder custom admin portal and is not the supported admin path

## LLM Provider Architecture

All LLM calls across the project are routed through a central client:

`adaptive-virtual-assistant-beta/ava_apps/core/services/llm_client.py`

This module provides:

- `get_client()` — returns a shared `OpenAI`-compatible client (works with both Ollama and OpenAI)
- `get_model()` — returns the model name for heavy tasks
- `get_fast_model()` — returns the model name for light/fast tasks (e.g., classification)
- `chat_completion(messages)` — convenience wrapper

The provider is controlled by the `LLM_PROVIDER` environment variable (`ollama` or `openai`).

Service files that use the LLM client:

| Service | Purpose |
|---------|--------|
| `answer_generation_service.py` | Chat answer generation |
| `knowledge_base_service.py` | Goal classification + KB generation (3 call sites) |
| `self_assessment_evaluation_service.py` | Self-assessment scoring |
| `dynamic_template_service.py` | Example text generation |
| `key_points_service.py` | Key point extraction |
| `evaluation_service.py` | Understanding check evaluation |

## Image Search Integration

Every chat answer is automatically enriched with relevant images:

1. **Wikimedia Commons** (primary) — free API, no key needed, searches for educational diagrams
2. **ChromaDB + OpenCLIP** (fallback) — local semantic image search over a pre-populated database

The image gallery appears as a `📷 Related Images` section below each chat answer. Images are clickable and open the full-size version in a new tab.

To populate the local ChromaDB fallback:

```bash
cd ../SER517_team5_industry/database
python seed_test_images.py
```

## Recommended Team Workflow

For new developers:

1. Clone the repo
2. Create a new empty MySQL database
3. Copy `.env.example` to `.env`
4. Install dependencies
5. Install Ollama and pull a model (`ollama pull llama3.2`)
6. Run `python manage.py migrate`
7. Run `python manage.py createsuperuser`
8. Start Ollama (`ollama serve`) and the Django server

That is the expected clean setup path for this Django project.
