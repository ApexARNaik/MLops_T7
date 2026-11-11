/**
 * Navigation Tabs Component
 * Controls active panels between Hobby Recommender, Answer Evaluator, Teacher Review, and Model Diagnostics.
 */
export class TabsComponent {
  constructor(navContainerId, onTabChange) {
    this.navContainer = document.getElementById(navContainerId);
    this.onTabChange = onTabChange;
    this.activeTab = 'hobby-learning';
    this.tabs = [
      { id: 'hobby-learning', label: '1. Hobby Recommender (Track A)' },
      { id: 'answer-feedback', label: '2. Answer Feedback & OCR (Track B)' },
      { id: 'teacher-review', label: '3. Teacher Approval & Override' },
      { id: 'model-inspector', label: '4. Model Inspector & Diagnostics' },
    ];
  }

  render() {
    if (!this.navContainer) return;
    this.navContainer.innerHTML = `
      <div class="tabs-nav">
        ${this.tabs
          .map(
            (tab) => `
          <button class="tab-btn ${tab.id === this.activeTab ? 'active' : ''}" data-tab="${tab.id}">
            ${tab.label}
          </button>
        `
          )
          .join('')}
      </div>
    `;

    this.navContainer.querySelectorAll('.tab-btn').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const tabId = e.currentTarget.getAttribute('data-tab');
        this.setActive(tabId);
      });
    });
  }

  setActive(tabId) {
    this.activeTab = tabId;
    this.navContainer.querySelectorAll('.tab-btn').forEach((btn) => {
      btn.classList.toggle('active', btn.getAttribute('data-tab') === tabId);
    });

    document.querySelectorAll('.tab-panel').forEach((panel) => {
      panel.classList.toggle('active', panel.id === `tab-${tabId}`);
    });

    if (typeof this.onTabChange === 'function') {
      this.onTabChange(tabId);
    }
  }
}

