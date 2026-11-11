/**
 * Header & System Status Component
 * Displays application branding, API readiness indicator, and current request tracking ID.
 */
export class HeaderComponent {
  constructor(containerId, options = {}) {
    this.container = document.getElementById(containerId);
    this.apiBaseUrl = options.apiBaseUrl || 'http://127.0.0.1:8000';
    this.status = { ready: false, text: 'Connecting...', models: [] };
  }

  render() {
    if (!this.container) return;
    this.container.innerHTML = `
      <nav class="navbar">
        <div class="container nav-content">
          <div class="nav-brand">
            <div class="brand-icon">AI</div>
            <div class="brand-text">
              <h1>Student Study Assistant</h1>
              <p>MLOps Serving Console & Demonstration UI</p>
            </div>
          </div>
          <div class="nav-status">
            <span id="system-status-badge" class="status-badge ${this.status.ready ? 'ready' : 'error'}">
              <span class="status-dot"></span>
              <span id="system-status-text">${this.status.text}</span>
            </span>
          </div>
        </div>
      </nav>
    `;
  }

  async checkHealth() {
    const badge = document.getElementById('system-status-badge');
    const textEl = document.getElementById('system-status-text');
    try {
      const res = await fetch(`${this.apiBaseUrl}/ready`);
      if (res.ok) {
        const data = await res.json();
        this.status = { ready: true, text: `Models Ready (${data.models ? data.models.length : 2})`, models: data.models };
        if (badge && textEl) {
          badge.className = 'status-badge ready';
          textEl.textContent = this.status.text;
        }
      } else {
        throw new Error('API not ready');
      }
    } catch (e) {
      this.status = { ready: false, text: 'API Offline (Local Mode)', models: [] };
      if (badge && textEl) {
        badge.className = 'status-badge error';
        textEl.textContent = this.status.text;
      }
    }
  }
}

