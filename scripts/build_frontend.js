/**
 * Zero-dependency static build script for Vercel / Production deployment.
 * Copies app/ assets to a target output directory (default: 'dist') and
 * injects environment-based API base URL from VITE_API_BASE_URL or API_BASE_URL.
 */

const fs = require('fs');
const path = require('path');

const repoRoot = path.resolve(__dirname, '..');
const srcDir = path.join(repoRoot, 'app');
const outDir = process.env.BUILD_OUTPUT_DIR || path.join(repoRoot, 'dist');

function copyDirRecursive(src, dest) {
  if (!fs.existsSync(dest)) {
    fs.mkdirSync(dest, { recursive: true });
  }

  const entries = fs.readdirSync(src, { withFileTypes: true });
  for (const entry of entries) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);

    if (entry.isDirectory()) {
      copyDirRecursive(srcPath, destPath);
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  }
}

console.log(`Building frontend from: ${srcDir}`);
console.log(`Output directory:        ${outDir}`);

// 1. Copy app/ contents to outDir
copyDirRecursive(srcDir, outDir);

// 2. Resolve production API Base URL
const rawApiUrl = process.env.VITE_API_BASE_URL || process.env.API_BASE_URL || '';
const apiUrl = rawApiUrl.trim().replace(/\/+$/, '');

if (apiUrl) {
  console.log(`Configuring production API_BASE_URL: ${apiUrl}`);
} else {
  console.warn('WARNING: Neither VITE_API_BASE_URL nor API_BASE_URL was provided. Defaulting to http://127.0.0.1:8000');
}

// 3. Write injected config.js in output directory
const configContent = `// Auto-generated during static frontend build
window.API_BASE_URL = '${apiUrl || "http://127.0.0.1:8000"}';
window.DEMO_MODE = false;
`;

fs.writeFileSync(path.join(outDir, 'config.js'), configContent, 'utf-8');
console.log('Frontend build completed successfully.');
