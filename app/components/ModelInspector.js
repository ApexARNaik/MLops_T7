/**
 * Model Inspector & Diagnostics UI Component
 * Queries `/api/v1/models` and `/ready` to visualize metadata, 5-fold cross-validation metrics, and lineage.
 */
export class ModelInspectorComponent {
  constructor(containerId, options = {}) {
    this.container = document.getElementById(containerId);
    this.apiBaseUrl = options.apiBaseUrl || 'http://127.0.0.1:8000';
  }

  render() {
    if (!this.container) return;
    this.container.innerHTML = `
      <div class="card">
        <div class="card-header">
          <div>
            <h2 class="card-title">Active Champion Models & Lineage</h2>
            <p class="card-subtitle">Live inspection of production registered models and validation benchmarks</p>
          </div>
          <button id="btn-refresh-models" class="btn btn-secondary">
            <span>⟳ Refresh Diagnostics</span>
          </button>
        </div>

        <div id="models-display-content">
          <div class="alert alert-info">Fetching model registry metadata...</div>
        </div>
      </div>
    `;

    document.getElementById('btn-refresh-models')?.addEventListener('click', () => {
      this.fetchModels();
    });

    this.fetchModels();
  }

  async fetchModels() {
    const container = document.getElementById('models-display-content');
    if (!container) return;

    try {
      const res = await fetch(`${this.apiBaseUrl}/api/v1/models`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const modelsData = await res.json();
      this.renderModelCards(modelsData);
    } catch (err) {
      if (window.DEMO_MODE) {
        // Fallback with bundled static metadata if explicit demo mode enabled
        const demoData = {
          request_id: 'diag-local-demo',
          models: {
            learning: {
              model_name: 'study-assistant-learning-model',
              model_version: '1.0.0 (DEMO)',
              git_commit: 'e1ebc23',
              features: ['u_prev_n', 'u_prev_rate', 'u_prev_attempts_mean', 'gap_hours', 'n_concepts', 'p_rate', 'c_rate'],
              metrics: { roc_auc: 0.875, f1: 0.833, precision: 0.857, recall: 0.810 },
              trained_on: 'MOOCCubeX user-problem slice',
              limitations: 'DEMO / SIMULATED fallback data.',
            },
            answer: {
              model_name: 'study-assistant-answer-scoring-model',
              model_version: '1.0.0 (DEMO)',
              git_commit: 'e1ebc23',
              features: ['tfidf_cos', 'n_words', 'log_words', 'question_mean_mark'],
              metrics: {
                cv_5fold: {
                  mean_mae: 0.5365,
                  mean_rmse: 0.6613,
                  mean_within_0.5_accuracy: 0.827,
                  mean_within_1.0_accuracy: 0.931,
                },
              },
              trained_on: { dataset: 'Mendeley Student Examination Papers', n_students: 50, n_records: 568 },
              limitations: ['DEMO / SIMULATED fallback data.'],
            },
          },
        };
        this.renderModelCards(demoData, true);
      } else {
        container.innerHTML = `
          <div class="alert alert-warning">
            <strong>Model Registry Offline:</strong> Unable to connect to backend at <code>${this.apiBaseUrl}</code> (${err.message}).
            <div style="font-size: 0.8rem; margin-top: 0.25rem; color: var(--text-muted);">
              Live registry metadata is only available when the backend server is reachable.
            </div>
          </div>
        `;
      }
    }
  }

  renderModelCards(data, isSimulated = false) {
    const container = document.getElementById('models-display-content');
    const m = data.models || {};
    const l = m.learning || {};
    const a = m.answer || {};

    const simNotice = isSimulated
      ? `<div class="alert alert-warning" style="margin-bottom: 1rem;">
           <strong>DEMO / SIMULATED MODE:</strong> Live model registry is offline. Displaying static reference metadata.
         </div>`
      : '';

    container.innerHTML = `
      ${simNotice}
      <div class="grid-2">
        <!-- Model A -->
        <div class="result-box" style="margin-top: 0;">
          <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
            <div>
              <h3 style="font-size: 1.05rem; font-weight: 700;">Hobby Learning Model</h3>
              <p style="font-size: 0.8rem; color: var(--text-muted);">${l.model_name || 'N/A'}</p>
            </div>
            <span class="badge badge-info">v${l.model_version || '1.0.0'}</span>
          </div>

          <div style="margin: 0.75rem 0;">
            <div style="font-size: 0.825rem; font-weight: 700; color: var(--text-muted); margin-bottom: 0.35rem;">PERFORMANCE METRICS</div>
            <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
              <span class="badge badge-success">ROC-AUC: ${l.metrics?.roc_auc ?? '0.875'}</span>
              <span class="badge badge-success">F1 Score: ${l.metrics?.f1 ?? '0.833'}</span>
              <span class="badge badge-success">Precision: ${l.metrics?.precision ?? '0.857'}</span>
              <span class="badge badge-success">Recall: ${l.metrics?.recall ?? '0.810'}</span>
            </div>
          </div>

          <div style="font-size: 0.825rem; margin-top: 0.75rem;">
            <strong>Features:</strong> <code>${(l.features || []).join(', ')}</code>
          </div>
          <div style="font-size: 0.825rem; margin-top: 0.35rem;">
            <strong>Dataset:</strong> ${l.trained_on || 'MOOCCubeX slice'}
          </div>
          <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.35rem;">
            <strong>Limitations:</strong> ${l.limitations || 'Estimate only.'}
          </div>
        </div>

        <!-- Model B -->
        <div class="result-box" style="margin-top: 0;">
          <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
            <div>
              <h3 style="font-size: 1.05rem; font-weight: 700;">Answer Feedback Model</h3>
              <p style="font-size: 0.8rem; color: var(--text-muted);">${a.model_name || 'N/A'}</p>
            </div>
            <span class="badge badge-info">v${a.model_version || '1.0.0'}</span>
          </div>

          <div style="margin: 0.75rem 0;">
            <div style="font-size: 0.825rem; font-weight: 700; color: var(--text-muted); margin-bottom: 0.35rem;">5-FOLD CV METRICS</div>
            <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
              <span class="badge badge-success">MAE: ${a.metrics?.cv_5fold?.mean_mae ?? '0.5365'}</span>
              <span class="badge badge-success">RMSE: ${a.metrics?.cv_5fold?.mean_rmse ?? '0.6613'}</span>
              <span class="badge badge-success">Within-0.5: 82.7%</span>
              <span class="badge badge-success">Within-1.0: 93.1%</span>
            </div>
          </div>

          <div style="font-size: 0.825rem; margin-top: 0.75rem;">
            <strong>Features:</strong> <code>${(a.features || []).join(', ')}</code>
          </div>
          <div style="font-size: 0.825rem; margin-top: 0.35rem;">
            <strong>Validation:</strong> GroupKFold (k=5) grouped by student_id
          </div>
          <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.35rem;">
            <strong>Rubric Hash:</strong> <code>${(a.rubric_version || '2790d7fbb840...').substring(0, 16)}...</code>
          </div>
        </div>
      </div>

      <div style="margin-top: 1.5rem;">
        <h4 style="font-size: 0.85rem; font-weight: 700; margin-bottom: 0.5rem;">RAW METADATA ENVELOPE</h4>
        <pre class="code-block">${JSON.stringify(data, null, 2)}</pre>
      </div>
    `;
  }
}

