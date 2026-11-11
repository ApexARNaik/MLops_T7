/**
 * Main Application Orchestrator
 * Bootstraps and coordinates all UI components.
 */
import { HeaderComponent } from './components/Header.js';
import { TabsComponent } from './components/Tabs.js';
import { HobbyRecommenderComponent } from './components/HobbyRecommender.js';
import { AnswerEvaluatorComponent } from './components/AnswerEvaluator.js';
import { TeacherReviewComponent } from './components/TeacherReview.js';
import { ModelInspectorComponent } from './components/ModelInspector.js';

document.addEventListener('DOMContentLoaded', () => {
  const API_BASE_URL = window.API_BASE_URL || 'http://127.0.0.1:8000';

  // 1. Initialize Header
  const header = new HeaderComponent('header-mount', { apiBaseUrl: API_BASE_URL });
  header.render();
  header.checkHealth();
  setInterval(() => header.checkHealth(), 15000);

  // 2. Initialize Subcomponents
  const hobbyRecommender = new HobbyRecommenderComponent('hobby-recommender-mount', {
    apiBaseUrl: API_BASE_URL,
  });
  hobbyRecommender.render();

  const teacherReview = new TeacherReviewComponent('teacher-review-mount', {
    apiBaseUrl: API_BASE_URL,
  });
  teacherReview.render();

  const answerEvaluator = new AnswerEvaluatorComponent('answer-evaluator-mount', {
    apiBaseUrl: API_BASE_URL,
    onSendToReview: (data) => {
      tabs.setActive('teacher-review');
      teacherReview.setData(data);
    },
  });
  answerEvaluator.render();

  const modelInspector = new ModelInspectorComponent('model-inspector-mount', {
    apiBaseUrl: API_BASE_URL,
  });
  modelInspector.render();

  // 3. Initialize Tabs
  const tabs = new TabsComponent('tabs-mount', (activeTab) => {
    if (activeTab === 'model-inspector') {
      modelInspector.fetchModels();
    }
  });
  tabs.render();
});

