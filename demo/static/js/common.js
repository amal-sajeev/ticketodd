/* ==========================================================================
   PR&DW Grievance Portal - Common JS
   ========================================================================== */

const API = '';

const ODISHA_DISTRICTS = [
  "Angul", "Balangir", "Balasore", "Bargarh", "Bhadrak", "Boudh", "Cuttack", "Deogarh",
  "Dhenkanal", "Gajapati", "Ganjam", "Jagatsinghpur", "Jajpur", "Jharsuguda", "Kalahandi",
  "Kandhamal", "Kendrapara", "Kendujhar", "Khordha", "Koraput", "Malkangiri", "Mayurbhanj",
  "Nabarangpur", "Nayagarh", "Nuapada", "Puri", "Rayagada", "Sambalpur", "Subarnapur", "Sundargarh"
];

const DEPARTMENTS = [
  "panchayati_raj", "rural_water_supply", "mgnregs", "rural_housing",
  "rural_livelihoods", "sanitation", "infrastructure", "general"
];

// --- Auth ---
function getToken() { return localStorage.getItem('token'); }
function getUser() { try { return JSON.parse(localStorage.getItem('user')); } catch { return null; } }
function setAuth(token, user) { localStorage.setItem('token', token); localStorage.setItem('user', JSON.stringify(user)); }
function clearAuth() { localStorage.removeItem('token'); localStorage.removeItem('user'); }

function isTokenExpired() {
  const token = getToken();
  if (!token) return true;
  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    return !payload.exp || (payload.exp * 1000) < Date.now();
  } catch { return true; }
}

function isLoggedIn() { return !!getToken() && !isTokenExpired(); }

function requireAuth() {
  if (!getToken()) { window.location.href = '/login'; return false; }
  if (isTokenExpired()) {
    clearAuth();
    showToast('Session expired — please log in again', 'error');
    setTimeout(() => { window.location.href = '/login'; }, 1500);
    return false;
  }
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
    if (res.status === 401) {
      const err = await res.json().catch(() => ({}));
      // Auth endpoints return 401 for wrong credentials — don't treat as session expiry
      if (endpoint.startsWith('/auth/')) {
        throw new Error(err.detail || 'Authentication failed');
      }
      clearAuth();
      showToast('Session expired — please log in again', 'error');
      setTimeout(() => { window.location.href = '/login'; }, 1500);
      throw new Error('Session expired');
    }
    if (res.status === 429) {
      throw new Error('Too many requests — please wait a moment and try again');
    }
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }
    return await res.json();
  } catch (e) {
    if (e.message?.includes('Request failed') || e.message === 'Session expired'
      || e.message?.includes('Authentication') || e.message?.includes('Too many')) throw e;
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
    if (res.status === 401) {
      const err = await res.json().catch(() => ({}));
      if (endpoint.startsWith('/auth/')) {
        throw new Error(err.detail || 'Authentication failed');
      }
      clearAuth();
      showToast('Session expired — please log in again', 'error');
      setTimeout(() => { window.location.href = '/login'; }, 1500);
      throw new Error('Session expired');
    }
    if (res.status === 429) {
      throw new Error('Too many requests — please wait a moment and try again');
    }
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }
    return await res.json();
  } catch (e) {
    if (e.message?.includes('Request failed') || e.message === 'Session expired'
      || e.message?.includes('Authentication') || e.message?.includes('Too many')) throw e;
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
      return DOMPurify.sanitize(raw, { ALLOWED_TAGS: ['p', 'br', 'strong', 'em', 'ul', 'ol', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a', 'code', 'pre', 'blockquote', 'table', 'thead', 'tbody', 'tr', 'th', 'td', 'hr', 'span', 'div'], ALLOWED_ATTR: ['href', 'target', 'class'] });
    }
    return raw;
  }
  return escapeHtml(text).replace(/\n/g, '<br>');
}

// --- Toast ---
function showToast(message, type = 'success') {
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  const iconName = type === 'success' ? 'check_circle' : 'error';
  el.innerHTML = `<span class="icon filled" style="font-size:18px;">${iconName}</span> ${escapeHtml(message)}`;
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
      <a href="/community">Community</a>
      <a href="/profile">Profile</a>`;
  } else if (user.role === 'admin') {
    links = `
      <a href="/officer-dashboard">Dashboard</a>
      <a href="/queue">Queue</a>
      <a href="/knowledge">Knowledge</a>
      <a href="/analytics-view">Analytics</a>
      <a href="/admin/reports">Reports</a>
      <a href="/admin">Admin</a>
      <a href="/community">Community</a>
      <a href="/profile">Profile</a>`;
  } else {
    links = `
      <a href="/officer-dashboard">Dashboard</a>
      <a href="/queue">Queue</a>
      <a href="/knowledge">Knowledge</a>
      <a href="/analytics-view">Analytics</a>
      <a href="/community">Community</a>
      <a href="/profile">Profile</a>`;
  }

  const bellHtml = (user.role === 'admin') ? `
    <div class="notif-bell-wrap" id="notifBellWrap">
      <button class="notif-bell" id="notifBell" onclick="toggleNotifDropdown()" title="SLA Alerts">
        <span class="icon">notifications</span><span class="notif-badge" id="notifBadge" style="display:none;">0</span>
      </button>
      <div class="notif-dropdown" id="notifDropdown">
        <div class="notif-dropdown-header"><span class="icon" style="font-size:18px;color:var(--warning);">warning</span> SLA Deadline Alerts</div>
        <div class="notif-dropdown-body" id="notifDropdownBody">
          <div style="padding:1rem;text-align:center;opacity:0.6;">Loading…</div>
        </div>
        <a href="/admin" class="notif-dropdown-footer">View All in Admin Panel</a>
      </div>
    </div>` : '';

  nav.innerHTML = `
    <a class="logo" href="${user.role === 'citizen' ? '/dashboard' : '/officer-dashboard'}">
      <span class="icon filled">account_balance</span> PR&DW Portal
    </a>
    <div class="nav-links">${links}</div>
    <div class="nav-user">
      ${bellHtml}
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

// --- Notification Bell (Admin) ---
let _notifDropdownOpen = false;
let _notifPollTimer = null;

function toggleNotifDropdown() {
  const dd = document.getElementById('notifDropdown');
  if (!dd) return;
  _notifDropdownOpen = !_notifDropdownOpen;
  dd.classList.toggle('open', _notifDropdownOpen);
  if (_notifDropdownOpen) fetchNotifAlerts();
}

function closeNotifDropdown(e) {
  if (_notifDropdownOpen && !e.target.closest('.notif-bell-wrap')) {
    _notifDropdownOpen = false;
    const dd = document.getElementById('notifDropdown');
    if (dd) dd.classList.remove('open');
  }
}

async function fetchNotifAlerts() {
  try {
    const data = await api('GET', '/admin/deadline-alerts');
    const badge = document.getElementById('notifBadge');
    const bell = document.getElementById('notifBell');
    if (badge) {
      if (data.total_alerts > 0) {
        badge.textContent = data.total_alerts > 99 ? '99+' : data.total_alerts;
        badge.style.display = 'flex';
        if (bell) bell.classList.add('has-alerts');
      } else {
        badge.style.display = 'none';
        if (bell) bell.classList.remove('has-alerts');
      }
    }
    // Render dropdown body
    const body = document.getElementById('notifDropdownBody');
    if (!body) return;
    if (data.total_alerts === 0) {
      body.innerHTML = '<div style="padding:1.25rem;text-align:center;color:var(--success);font-weight:600;"><span class="icon filled" style="font-size:20px;vertical-align:middle;">check_circle</span> All clear — no alerts</div>';
      return;
    }
    let html = '';
    const items = [
      ...data.breached.map(g => ({ ...g, sev: 'breached', icon: '<span class="icon filled" style="color:var(--error);font-size:18px;">error</span>' })),
      ...data.critical.map(g => ({ ...g, sev: 'critical', icon: '<span class="icon" style="color:var(--warning);font-size:18px;">schedule</span>' })),
      ...data.warning.map(g => ({ ...g, sev: 'warning', icon: '<span class="icon" style="color:#FDD835;font-size:18px;">bolt</span>' })),
    ].slice(0, 8);
    items.forEach(g => {
      const timeText = _notifCountdown(g.sla_deadline, g.sev);
      html += `<a href="/grievance-detail?id=${encodeURIComponent(g.id)}" class="notif-item notif-${g.sev}">
        <span class="notif-item-icon">${g.icon}</span>
        <span class="notif-item-body">
          <span class="notif-item-title">${escapeHtml(g.title)}</span>
          <span class="notif-item-meta">${escapeHtml(g.tracking_number)} · ${timeText}</span>
        </span>
      </a>`;
    });
    if (data.total_alerts > 8) {
      html += `<div style="padding:0.5rem 1rem;text-align:center;font-size:0.82rem;opacity:0.7;">+${data.total_alerts - 8} more…</div>`;
    }
    body.innerHTML = html;
  } catch (e) {
    const body = document.getElementById('notifDropdownBody');
    if (body) body.innerHTML = '<div style="padding:1rem;text-align:center;color:#C62828;">Failed to load alerts</div>';
  }
}

function _notifCountdown(iso, sev) {
  if (!iso) return '';
  const diff = new Date(iso) - new Date();
  if (diff <= 0) {
    const h = Math.floor(Math.abs(diff) / 3600000);
    const d = Math.floor(h / 24);
    return d > 0 ? `${d}d ${h % 24}h overdue` : `${h}h overdue`;
  }
  const h = Math.floor(diff / 3600000);
  const m = Math.floor((diff % 3600000) / 60000);
  return h > 0 ? `${h}h ${m}m left` : `${m}m left`;
}

function startNotifPolling() {
  const user = getUser();
  if (!user || user.role !== 'admin') return;
  // Initial fetch (silent, just badge count)
  fetchNotifAlerts();
  // Poll every 60 seconds
  _notifPollTimer = setInterval(fetchNotifAlerts, 60000);
}

// --- Init nav on load ---
document.addEventListener('DOMContentLoaded', () => {
  renderNav();
  startNotifPolling();
  document.addEventListener('click', closeNotifDropdown);
});
