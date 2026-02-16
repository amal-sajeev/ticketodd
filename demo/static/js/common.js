/* ==========================================================================
   PR&DW Grievance Portal - Common JS
   ========================================================================== */

const API = '';

const ODISHA_DISTRICTS = [
  "Angul","Balangir","Balasore","Bargarh","Bhadrak","Boudh","Cuttack","Deogarh",
  "Dhenkanal","Gajapati","Ganjam","Jagatsinghpur","Jajpur","Jharsuguda","Kalahandi",
  "Kandhamal","Kendrapara","Kendujhar","Khordha","Koraput","Malkangiri","Mayurbhanj",
  "Nabarangpur","Nayagarh","Nuapada","Puri","Rayagada","Sambalpur","Subarnapur","Sundargarh"
];

const DEPARTMENTS = [
  "panchayati_raj","rural_water_supply","mgnregs","rural_housing",
  "rural_livelihoods","sanitation","infrastructure","general"
];

// --- Auth ---
function getToken() { return localStorage.getItem('token'); }
function getUser() { try { return JSON.parse(localStorage.getItem('user')); } catch { return null; } }
function setAuth(token, user) { localStorage.setItem('token', token); localStorage.setItem('user', JSON.stringify(user)); }
function clearAuth() { localStorage.removeItem('token'); localStorage.removeItem('user'); }
function isLoggedIn() { return !!getToken(); }

function requireAuth() {
  if (!isLoggedIn()) { window.location.href = '/login'; return false; }
  return true;
}
function requireRole(role) {
  const u = getUser();
  if (!u || u.role !== role) {
    window.location.href = u?.role === 'citizen' ? '/dashboard' : '/officer-dashboard';
    return false;
  }
  return true;
}

// --- API Client ---
async function api(method, endpoint, body = null) {
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json' },
  };
  const token = getToken();
  if (token) opts.headers['Authorization'] = `Bearer ${token}`;
  if (body) opts.body = JSON.stringify(body);
  try {
    const res = await fetch(`${API}${endpoint}`, opts);
    if (res.status === 401) { clearAuth(); window.location.href = '/login'; return null; }
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }
    return await res.json();
  } catch (e) {
    if (e.message?.includes('Request failed')) throw e;
    console.error('API Error:', e);
    throw e;
  }
}

// --- FormData API helper (for file uploads) ---
async function apiFormData(method, endpoint, formData) {
  const opts = { method, body: formData, headers: {} };
  const token = getToken();
  if (token) opts.headers['Authorization'] = `Bearer ${token}`;
  try {
    const res = await fetch(`${API}${endpoint}`, opts);
    if (res.status === 401) { clearAuth(); window.location.href = '/login'; return null; }
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }
    return await res.json();
  } catch (e) {
    if (e.message?.includes('Request failed')) throw e;
    console.error('API Error:', e);
    throw e;
  }
}


// --- Utilities ---
function deptLabel(d) {
  const labels = {
    panchayati_raj: 'Panchayati Raj', rural_water_supply: 'Rural Water Supply',
    mgnregs: 'MGNREGS', rural_housing: 'Rural Housing',
    rural_livelihoods: 'Rural Livelihoods', sanitation: 'Sanitation',
    infrastructure: 'Infrastructure', general: 'General'
  };
  return labels[d] || (d || 'general').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function formatDate(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    return d.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
  } catch { return iso; }
}

function statusBadge(status) {
  const safe = escapeHtml(status || '');
  const s = (status || '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
  return `<span class="badge badge-${safe}">${escapeHtml(s)}</span>`;
}

function priorityBadge(p) {
  const safe = escapeHtml(p || '');
  const label = (p || '').charAt(0).toUpperCase() + (p || '').slice(1);
  return `<span class="badge badge-priority-${safe}">${escapeHtml(label)}</span>`;
}

function deptBadge(d) { return `<span class="badge badge-dept">${escapeHtml(deptLabel(d))}</span>`; }

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function escapeJs(text) {
  if (!text) return '';
  return String(text).replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/"/g, '\\"').replace(/\n/g, '\\n').replace(/\r/g, '\\r');
}

// --- Markdown Rendering (sanitized) ---
function renderMarkdown(text) {
  if (!text) return '';
  if (typeof marked !== 'undefined') {
    marked.setOptions({ breaks: true, gfm: true });
    const raw = marked.parse(text);
    if (typeof DOMPurify !== 'undefined') {
      return DOMPurify.sanitize(raw, { ALLOWED_TAGS: ['p','br','strong','em','ul','ol','li','h1','h2','h3','h4','h5','h6','a','code','pre','blockquote','table','thead','tbody','tr','th','td','hr','span','div'], ALLOWED_ATTR: ['href','target','class'] });
    }
    return raw;
  }
  return escapeHtml(text).replace(/\n/g, '<br>');
}

// --- Toast ---
function showToast(message, type = 'success') {
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.textContent = message;
  document.body.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 300); }, 3000);
}

// --- Nav ---
function renderNav() {
  const nav = document.getElementById('navbar');
  if (!nav) return;
  const user = getUser();
  if (!user) { nav.innerHTML = ''; return; }

  let links = '';
  if (user.role === 'citizen') {
    links = `
      <a href="/dashboard">Dashboard</a>
      <a href="/file-grievance">File Grievance</a>
      <a href="/track">Track</a>
      <a href="/chatbot">Chatbot</a>
      <a href="/schemes">Schemes</a>
      <a href="/community">Community</a>`;
  } else if (user.role === 'admin') {
    links = `
      <a href="/officer-dashboard">Dashboard</a>
      <a href="/queue">Queue</a>
      <a href="/knowledge">Knowledge</a>
      <a href="/analytics-view">Analytics</a>
      <a href="/admin">Admin</a>
      <a href="/community">Community</a>`;
  } else {
    links = `
      <a href="/officer-dashboard">Dashboard</a>
      <a href="/queue">Queue</a>
      <a href="/knowledge">Knowledge</a>
      <a href="/analytics-view">Analytics</a>
      <a href="/community">Community</a>`;
  }

  nav.innerHTML = `
    <a class="logo" href="${user.role === 'citizen' ? '/dashboard' : '/officer-dashboard'}">
      <span>🏛️</span> PR&DW Portal
    </a>
    <div class="nav-links">${links}</div>
    <div class="nav-user">
      <span class="user-badge">${escapeHtml(user.full_name)} (${user.role})</span>
      <button class="btn-logout" onclick="logout()">Logout</button>
    </div>`;

  // Highlight active
  const path = window.location.pathname;
  nav.querySelectorAll('.nav-links a').forEach(a => {
    if (a.getAttribute('href') === path) a.classList.add('active');
  });
}

function logout() {
  clearAuth();
  window.location.href = '/login';
}

// --- Loading ---
function showLoading(container) {
  if (typeof container === 'string') container = document.getElementById(container);
  if (container) container.innerHTML = '<div class="spinner"><span></span><span></span><span></span></div>';
}

// --- District options ---
function districtOptions(selected) {
  return '<option value="">Select District</option>' +
    ODISHA_DISTRICTS.map(d => `<option value="${d}"${d === selected ? ' selected' : ''}>${d}</option>`).join('');
}

function deptOptions(selected, includeAll = false) {
  let html = includeAll ? '<option value="">All Departments</option>' : '';
  html += DEPARTMENTS.map(d => `<option value="${d}"${d === selected ? ' selected' : ''}>${deptLabel(d)}</option>`).join('');
  return html;
}

// --- Init nav on load ---
document.addEventListener('DOMContentLoaded', () => {
  renderNav();
});
