/**
 * Answer Feedback & Scoring UI Component
 * Implements question selection, OCR confidence tuning, evaluation breakdown, and teacher handoff.
 */
export class AnswerEvaluatorComponent {
  constructor(containerId, options = {}) {
    this.container = document.getElementById(containerId);
    this.apiBaseUrl = options.apiBaseUrl || 'http://127.0.0.1:8000';
    this.onSendToReview = options.onSendToReview || null;

    this.sampleQuestions = {
      Q21: {
        text: 'Define Supervised Learning.',
        reference: 'A type of ML where the model is trained on labeled data with known inputs and outputs.',
        max_mark: 2.0,
        full_sample: 'Supervised learning is where the model is trained on labeled data with known inputs and outputs.',
        partial_sample: 'It is a form of learning that uses data to predict things.',
      },
      Q22: {
        text: 'What is happening to M1? (Overfitting vs Generalization)',
        reference: 'Overfitting; the model memorized the training data but failed to generalize.',
        max_mark: 2.0,
        full_sample: 'The model suffers from overfitting because it memorized the training data and failed to generalize to test data.',
        partial_sample: 'The model has high variance.',
      },
      Q23: {
        text: 'Explain why L2 regularization helps avoid overfitting.',
        reference: 'Penalizes large weights by adding the sum of squared coefficients to the loss function.',
        max_mark: 2.0,
        full_sample: 'L2 regularization adds a penalty on the sum of squared weights, shrinking coefficient magnitudes.',
        partial_sample: 'It makes the model smaller.',
      },
      Q24: {
        text: 'Distinguish between Precision and Recall.',
        reference: 'Precision is true positives over predicted positives; recall is true positives over all actual positives.',
        max_mark: 2.0,
        full_sample: 'Precision measures true positives out of all predicted positives, while recall measures true positives out of all actual ground truth positives.',
        partial_sample: 'Precision is how accurate you are, recall is finding things.',
      },
    };
  }

  render() {
    if (!this.container) return;
    this.container.innerHTML = `
      <div class="grid-2">
        <!-- Input Form -->
        <div class="card">
          <div class="card-header">
            <div>
              <h2 class="card-title">Answer Sheet Scoring Input</h2>
              <p class="card-subtitle">Evaluate short answers with rubric matching and OCR confidence</p>
            </div>
            <span class="badge badge-info">POST /api/v1/answer-feedback/evaluate</span>
          </div>

          <form id="answer-form">
            <div class="form-group">
              <label class="form-label" for="assessment_id">Assessment ID</label>
              <input type="text" id="assessment_id" class="form-input" value="exam-session-001" required />
            </div>

            <div class="form-group">
              <label class="form-label" for="eval_qid">Exam Question</label>
              <select id="eval_qid" class="form-select">
                <option value="Q21">Q21: Define Supervised Learning</option>
                <option value="Q22">Q22: Overfitting Analysis (M1)</option>
                <option value="Q23">Q23: L2 Regularization Benefits</option>
                <option value="Q24">Q24: Precision vs. Recall</option>
              </select>
            </div>

            <div id="question-prompt-box" class="alert alert-info" style="font-size: 0.85rem; margin-bottom: 1rem;">
              <!-- Dynamic prompt details injected here -->
            </div>

            <div class="form-group">
              <label class="form-label" for="answer_text">Student Answer Text (Segmented or OCR)</label>
              <textarea id="answer_text" class="form-textarea" placeholder="Enter student response text..."></textarea>
              <div class="quick-pills">
                <span class="form-hint" style="margin-right: 0.25rem;">Prefill Test:</span>
                <button type="button" class="pill-btn" id="btn-prefill-full">Full Score Answer</button>
                <button type="button" class="pill-btn" id="btn-prefill-partial">Partial Answer</button>
                <button type="button" class="pill-btn" id="btn-prefill-blank">Empty / Blank</button>
              </div>
            </div>

            <div class="form-group">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.35rem;">
                <label class="form-label" for="ocr_confidence" style="margin-bottom: 0;">OCR Confidence Score</label>
                <span id="ocr_confidence_display" class="badge badge-success">0.95</span>
              </div>
              <input type="range" id="ocr_confidence" min="0.0" max="1.0" step="0.05" value="0.95" style="width: 100%;" />
              <p class="form-hint">Values &lt; 0.60 trigger mandatory human review per safety protocol.</p>
            </div>

            <button type="submit" id="btn-evaluate" class="btn btn-primary" style="width: 100%;">
              <span>Evaluate Answer</span>
            </button>
          </form>
        </div>

        <!-- Result Card -->
        <div class="card">
          <div class="card-header">
            <div>
              <h2 class="card-title">Evaluation & Rubric Breakdown</h2>
              <p class="card-subtitle">AI-estimated marks and granular rubric point alignment</p>
            </div>
            <div id="eval-badges"></div>
          </div>

          <div id="eval-result-content">
            <div class="alert alert-info" style="margin-top: 0;">
              <span>Submit a student answer to view marks, matched points, and rubric feedback.</span>
            </div>
          </div>
        </div>
      </div>
    `;

    this.bindEvents();
    this.updateQuestionPrompt();
  }

  bindEvents() {
    const qidSelect = document.getElementById('eval_qid');
    if (qidSelect) {
      qidSelect.addEventListener('change', () => this.updateQuestionPrompt());
    }

    const ocrSlider = document.getElementById('ocr_confidence');
    const ocrDisplay = document.getElementById('ocr_confidence_display');
    if (ocrSlider && ocrDisplay) {
      ocrSlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        ocrDisplay.textContent = val.toFixed(2);
        if (val < 0.6) {
          ocrDisplay.className = 'badge badge-warning';
        } else {
          ocrDisplay.className = 'badge badge-success';
        }
      });
    }

    document.getElementById('btn-prefill-full')?.addEventListener('click', () => {
      const qid = document.getElementById('eval_qid').value;
      document.getElementById('answer_text').value = this.sampleQuestions[qid]?.full_sample || '';
    });

    document.getElementById('btn-prefill-partial')?.addEventListener('click', () => {
      const qid = document.getElementById('eval_qid').value;
      document.getElementById('answer_text').value = this.sampleQuestions[qid]?.partial_sample || '';
    });

    document.getElementById('btn-prefill-blank')?.addEventListener('click', () => {
      document.getElementById('answer_text').value = '';
    });

    document.getElementById('answer-form')?.addEventListener('submit', (e) => {
      e.preventDefault();
      this.submitEvaluation();
    });
  }

  updateQuestionPrompt() {
    const qid = document.getElementById('eval_qid').value;
    const promptBox = document.getElementById('question-prompt-box');
    const q = this.sampleQuestions[qid];
    if (q && promptBox) {
      promptBox.innerHTML = `
        <strong>${qid}:</strong> ${q.text}<br/>
        <span style="color: var(--text-muted); font-size: 0.8rem;">
          <strong>Target Concept:</strong> ${q.reference} (Max: ${q.max_mark.toFixed(1)} marks)
        </span>
      `;
      // default prefill to full
      document.getElementById('answer_text').value = q.full_sample;
    }
  }

  async submitEvaluation() {
    const btn = document.getElementById('btn-evaluate');
    const resultContent = document.getElementById('eval-result-content');
    const badgesContainer = document.getElementById('eval-badges');

    const qid = document.getElementById('eval_qid').value;
    const text = document.getElementById('answer_text').value;
    const ocrConf = parseFloat(document.getElementById('ocr_confidence').value);
    const assessmentId = document.getElementById('assessment_id').value.trim();

    const payload = {
      assessment_id: assessmentId,
      answers: { [qid]: text },
      ocr_confidence: { [qid]: ocrConf },
    };

    btn.disabled = true;
    btn.innerHTML = '<span>Scoring response...</span>';
    resultContent.innerHTML = '<div class="alert alert-info">Executing scoring model and rubric matcher...</div>';
    badgesContainer.innerHTML = '';

    try {
      const response = await fetch(`${this.apiBaseUrl}/api/v1/answer-feedback/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        const detailMsg = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
        throw new Error(detailMsg || `Server error ${response.status}`);
      }
      const resData = await response.json();
      this.renderEvaluationResult(resData, qid, assessmentId);
    } catch (err) {
      if (window.DEMO_MODE) {
        const demoData = this.simulateEvaluation(qid, text, ocrConf);
        this.renderEvaluationResult(demoData, qid, assessmentId, true);
      } else {
        resultContent.innerHTML = `
          <div class="alert alert-warning">
            <strong>Evaluation Error:</strong> ${err.message}
            <div style="font-size: 0.8rem; margin-top: 0.25rem; color: var(--text-muted);">
              Ensure backend server is running at <code>${this.apiBaseUrl}</code>.
            </div>
          </div>
        `;
      }
    } finally {
      btn.disabled = false;
      btn.innerHTML = '<span>Evaluate Answer</span>';
    }
  }

  simulateEvaluation(qid, text, ocrConf) {
    const words = (text || '').trim().split(/\s+/).filter(Boolean);
    const isBlank = words.length === 0;
    const isShort = words.length > 0 && words.length < 3;
    const isLowOcr = ocrConf < 0.6;
    const reviewReq = isBlank || isShort || isLowOcr;

    let mark = 0.0;
    let matched = [];
    let missing = [];

    if (qid === 'Q21') {
      if (text.toLowerCase().includes('labeled data')) matched.push('labeled data');
      else missing.push('labeled data');

      if (text.toLowerCase().includes('known input')) matched.push('known inputs and outputs');
      else missing.push('known inputs and outputs');

      mark = matched.length === 2 ? 2.0 : (matched.length === 1 ? 1.0 : 0.0);
    } else {
      mark = isBlank ? 0.0 : (words.length > 8 ? 2.0 : 1.0);
      matched = words.length > 5 ? ['core concept'] : [];
      missing = words.length <= 5 ? ['detailed justification'] : [];
    }

    return {
      request_id: 'local-eval-' + Math.random().toString(36).substring(2, 9),
      model_name: 'study-assistant-answer-scoring-model',
      model_version: '1.0.0',
      warnings: [],
      data: {
        results: [
          {
            qid: qid,
            estimated_mark: mark,
            max_mark: 2.0,
            approved_mark: null,
            status: 'pending_teacher_review',
            human_review_required: reviewReq,
            ocr_confidence: ocrConf,
            matched_points: matched,
            missing_points: missing,
            deduction_rationale: missing.length ? `Deducted for omission of: ${missing.join(', ')}` : 'Full marks awarded.',
            feedback_text: mark === 2.0 ? 'Excellent! All key rubric concepts were addressed.' : 'Partial credit awarded. Address missing elements for full marks.',
          },
        ],
      },
    };
  }

  renderEvaluationResult(res, qid, assessmentId, isSimulated = false) {
    const resultContent = document.getElementById('eval-result-content');
    const badgesContainer = document.getElementById('eval-badges');

    const item = res.data?.results?.[0] || {};
    const reviewReq = item.human_review_required;

    const simBadge = isSimulated
      ? '<span class="badge badge-warning" style="margin-left: 0.25rem;">DEMO / SIMULATED</span>'
      : '<span class="badge badge-success" style="margin-left: 0.25rem;">LIVE INFERENCE</span>';

    badgesContainer.innerHTML = `
      <span class="badge badge-info">${res.model_name || 'answer-model'} v${res.model_version || '1.0.0'}</span>
      ${simBadge}
    `;

    let reviewAlert = '';
    if (isSimulated) {
      reviewAlert += `
        <div class="alert alert-warning" style="margin-top: 0.5rem;">
          <strong>DEMO / SIMULATED MODE:</strong> Backend was unreachable. This is an offline mock, not actual model output.
        </div>
      `;
    }
    if (reviewReq) {
      reviewAlert = `
        <div class="alert alert-warning" style="margin-top: 0;">
          <strong>Human Review Required:</strong> This answer was flagged for manual teacher grading due to 
          ${item.ocr_confidence < 0.6 ? 'low OCR confidence (&lt; 0.60)' : 'insufficient word count or unattempted response'}.
        </div>
      `;
    }

    resultContent.innerHTML = `
      ${reviewAlert}

      <div class="result-box" style="display: flex; align-items: center; justify-content: space-between;">
        <div>
          <span style="font-size: 0.85rem; color: var(--text-muted); font-weight: 600;">ESTIMATED MARK</span>
          <div style="font-size: 1.75rem; font-weight: 800; color: var(--text-main);">
            ${(item.estimated_mark ?? 0).toFixed(1)} <span style="font-size: 1rem; color: var(--text-muted);">/ ${item.max_mark || 2.0}</span>
          </div>
        </div>
        <div>
          <span class="badge ${item.status === 'teacher_approved' ? 'badge-success' : 'badge-warning'}">
            ${item.status || 'pending_teacher_review'}
          </span>
        </div>
      </div>

      <div style="margin: 1.25rem 0;">
        <div style="font-size: 0.85rem; font-weight: 700; margin-bottom: 0.35rem;">Matched Rubric Points:</div>
        <div>
          ${(item.matched_points && item.matched_points.length) 
            ? item.matched_points.map((p) => `<span class="badge badge-success" style="margin-right: 0.25rem;">✓ ${p}</span>`).join('') 
            : '<span style="font-size: 0.85rem; color: var(--text-muted);">None matched</span>'}
        </div>
      </div>

      <div style="margin: 1.25rem 0;">
        <div style="font-size: 0.85rem; font-weight: 700; margin-bottom: 0.35rem;">Missing Elements:</div>
        <div>
          ${(item.missing_points && item.missing_points.length) 
            ? item.missing_points.map((p) => `<span class="badge badge-danger" style="margin-right: 0.25rem;">✗ ${p}</span>`).join('') 
            : '<span style="font-size: 0.85rem; color: var(--text-muted);">None missing</span>'}
        </div>
      </div>

      <div class="result-box">
        <div style="font-size: 0.8rem; font-weight: 700; color: var(--text-muted); margin-bottom: 0.25rem;">FEEDBACK & RATIONALE</div>
        <p style="font-size: 0.925rem; margin-bottom: 0.5rem;">${item.feedback_text || 'Assessment completed.'}</p>
        <p style="font-size: 0.85rem; color: var(--text-muted);">${item.deduction_rationale || ''}</p>
      </div>

      <div style="margin-top: 1.25rem; display: flex; gap: 0.75rem;">
        <button type="button" id="btn-handoff-review" class="btn btn-secondary" style="width: 100%;">
          <span>Forward to Teacher Review Panel →</span>
        </button>
      </div>
    `;

    document.getElementById('btn-handoff-review')?.addEventListener('click', () => {
      if (typeof this.onSendToReview === 'function') {
        this.onSendToReview({
          assessment_id: assessmentId,
          qid: qid,
          estimated_mark: item.estimated_mark ?? 0.0,
          max_mark: item.max_mark || 2.0,
        });
      }
    });
  }
}

