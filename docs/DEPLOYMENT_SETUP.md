# Deployment Setup Guide: Render Backend + Vercel Frontend

**Project**: AI-Powered Student Study Assistant  
**Backend**: FastAPI running on Render Web Services  
**Frontend**: SPA / Web App running on Vercel  
**CI/CD Pipeline**: GitHub Actions (`.github/workflows/ci.yml`)

---

## 1. Architecture Overview

```
                      +-----------------------------+
                      |       GitHub Repository     |
                      |   (main branch protected)   |
                      +--------------+--------------+
                                     |
               +---------------------+---------------------+
               |                                           |
               v                                           v
      +-----------------+                         +-----------------+
      |  GitHub Actions |                         |  GitHub Actions |
      |     (CI Run)    |                         |  (Smoke Test)   |
      +--------+--------+                         +--------+--------+
               |                                           |
        [All Gates Pass]                            [All Gates Pass]
               |                                           |
               v                                           v
    +----------------------+                    +----------------------+
    |    Render Backend    |                    |    Vercel Frontend   |
    |      (FastAPI)       | <== CORS Requests  |     (Static/SPA)     |
    |  https://*.render.com|                    |  https://*.vercel.app|
    +----------------------+                    +----------------------+
```

Production deployments are automated via GitHub integrations with Render and Vercel. Both platforms trigger automated deployments **only after changes are merged to the `main` branch**, which is protected by GitHub Actions quality gates.

---

## 2. Backend Deployment – Render

### 2.1 Repository Setup & Blueprint
Render natively reads [`render.yaml`](file:///c:/Users/Atul%20RN/Downloads/MLops_T7-1/render.yaml) located at the root of the repository.

1. Log into your [Render Dashboard](https://dashboard.render.com).
2. Click **New** > **Blueprint**.
3. Connect your GitHub repository: `ApexARNaik/MLops_T7`.
4. Render will parse `render.yaml` automatically and configure the `ai-study-assistant-api` web service.

Alternatively, to configure as an individual **Web Service** manually:
- **Environment**: Python
- **Root Directory**: `.` (repository root)
- **Build Command**: `pip install --no-cache-dir -r requirements-serving.txt`
- **Start Command**: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
- **Health Check Path**: `/health`
- **Auto-Deploy**: `Yes` (deploys automatically on push to `main`)

### 2.2 Environment Variables (Render)
Configure the following environment variables in the Render Dashboard (**Environment** tab):

| Variable | Recommended Production Value | Description |
| :--- | :--- | :--- |
| `PYTHON_VERSION` | `3.11.9` | Sets the Python runtime version. |
| `ALLOWED_ORIGINS` | `https://your-frontend.vercel.app` | Comma-separated list of allowed frontend origins for CORS. |
| `PORT` | *(Assigned dynamically by Render, default `8000`)* | Application listening port. |

> [!IMPORTANT]
> Never set `ALLOWED_ORIGINS="*"` in production. Explicitly provide your Vercel production and preview domains (e.g. `https://ai-study-assistant.vercel.app`).

### 2.3 Health Check Verification
Render will poll `/health` every few seconds during startup. Once the endpoint returns HTTP 200:
```json
{
  "status": "ok",
  "request_id": "c1f7a6b2-..."
}
```
Render marks the service as `Live` and directs incoming traffic to the new instance. Zero downtime is achieved through rolling updates.

---

## 3. Frontend Deployment – Vercel

### 3.1 Importing the Project to Vercel
1. Log into the [Vercel Dashboard](https://vercel.com).
2. Click **Add New** > **Project**.
3. Import your GitHub repository (`ApexARNaik/MLops_T7`).
4. In the project settings, configure:
   - **Root Directory**:
     - If your frontend files reside in a `frontend/` subdirectory, select `frontend`.
     - If your frontend files are in the repository root, leave it as `./`.
   - **Framework Preset**:
     - For Vite (React/Vue/Svelte): Select **Vite**.
     - For Next.js: Select **Next.js**.
     - For Create React App: Select **Create React App**.
     - For Static HTML: Select **Other**.
   - **Build Command**: `npm run build` (or leave default framework preset).
   - **Output Directory**: `dist` (for Vite) or `.next` (for Next.js) or `build` (for CRA).

### 3.2 Single-Page Application (SPA) Routing with `vercel.json`
The repository includes a root-level [`vercel.json`](file:///c:/Users/Atul%20RN/Downloads/MLops_T7-1/vercel.json) ensuring all client-side navigation paths rewrite to `index.html` without causing 404 errors on page refresh:
```json
{
  "version": 2,
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "/index.html"
    }
  ]
}
```

### 3.3 Environment Variables (Vercel)
Set the backend API connection URL in Vercel under **Settings** > **Environment Variables**:

| Variable | Framework | Value |
| :--- | :--- | :--- |
| `VITE_API_BASE_URL` | Vite | `https://ai-study-assistant-api.onrender.com` |
| `NEXT_PUBLIC_API_BASE_URL` | Next.js | `https://ai-study-assistant-api.onrender.com` |
| `REACT_APP_API_BASE_URL` | CRA | `https://ai-study-assistant-api.onrender.com` |

#### Recommended Frontend API Client Pattern:
```typescript
// src/api/client.ts
const API_BASE_URL = 
  import.meta.env?.VITE_API_BASE_URL || 
  process.env?.NEXT_PUBLIC_API_BASE_URL || 
  "http://localhost:8000";

export async function checkHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  return response.json();
}

export async function getHobbyRecommendation(payload: unknown) {
  const response = await fetch(`${API_BASE_URL}/api/v1/hobby-learning/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  return response.json();
}
```

---

## 4. GitHub Actions CI Pipeline

The CI workflow is defined in [`.github/workflows/ci.yml`](file:///c:/Users/Atul%20RN/Downloads/MLops_T7-1/.github/workflows/ci.yml).

### 4.1 Workflow Triggers
- **Pull Requests**: Every PR targeting `main`.
- **Pushes**: Every merge or direct push to `main`.

### 4.2 CI Jobs Overview
1. **`secret-scan`**: Scans commit history for exposed API keys, tokens, or credentials using `Gitleaks`.
2. **`backend-quality`**:
   - Sets up Python 3.11.
   - Installs serving and development dependencies.
   - Runs `ruff check` linting.
   - Runs `ruff format --check`.
   - Runs `pip-audit` for dependency vulnerabilities.
   - Executes the complete `pytest` test suite.
3. **`docker-smoke-test`**:
   - Builds the production Docker image.
   - Boots container and polls `/health` until HTTP 200.
   - Validates `/openapi.json` and readiness endpoint `/ready`.
4. **`frontend-quality`**:
   - Detects frontend directory and package setup dynamically.
   - Runs `npm ci`, `npm audit`, linting, tests, and production build.

---

## 5. GitHub Branch Protection Setup

To prevent unverified code from reaching production, configure branch protection in GitHub:

1. In GitHub, navigate to **Settings** > **Branches**.
2. Under **Branch protection rules**, click **Add rule**.
3. In **Branch name pattern**, enter: `main`.
4. Enable the following checkboxes:
   - [x] **Require a pull request before merging**
     - [x] Require approvals (at least 1 approval).
   - [x] **Require status checks to pass before merging**
     - [x] Require branches to be up to date before merging.
     - Select all CI jobs as required status checks:
       - `Security - Secret Scanning`
       - `Backend - Quality & Security`
       - `Backend - Docker Build & Smoke Test`
       - `Frontend - Quality & Build (Conditional)`
   - [x] **Do not allow bypassing the above settings**.
5. Click **Save changes**.

---

## 6. Deployment Workflow Summary

1. Developers work on feature branches (`feat/my-feature`).
2. A Pull Request is opened targeting `main`.
3. GitHub Actions triggers `.github/workflows/ci.yml`.
4. All quality gates execute in parallel.
5. If any check fails (linting, tests, secret detection, Docker smoke test), merging is blocked.
6. Once green and approved, the PR is merged into `main`.
7. Render and Vercel detect the new commit on `main`, execute production builds, and deploy with zero downtime.

