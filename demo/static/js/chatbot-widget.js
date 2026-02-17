/* ==========================================================================
   Floating Chatbot Widget
   Provides the same chat functionality as /chatbot in a persistent
   floating panel that stays across page navigations.
   ========================================================================== */

(function () {
  'use strict';

  const WIDGET_STORAGE_KEY = 'chatWidgetState';
  const TRANSFER_KEY = 'chatbot_transfer';
  const BOT_NAME = 'Seva';

  let chatHistory = [];
  let panelOpen = false;
  let unreadCount = 0;
  let wPendingFiles = [];
  let wUploadedAttachmentIds = [];

  const W_ALLOWED_TYPES = ['image/', 'audio/', 'video/'];
  const W_MAX_FILE_SIZE = 50 * 1024 * 1024;
  const W_MAX_FILES = 5;

  /* --- DOM references (set after injection) --- */
  let fab, badge, panel, messagesEl, inputEl, sendBtn, langSelect, wFileInput, wFilePreview;

  /* === Lifecycle ========================================================= */

  document.addEventListener('DOMContentLoaded', () => {
    if (!shouldShowWidget()) return;
    injectWidget();
    restoreState();
  });

  function shouldShowWidget() {
    if (window.location.pathname === '/chatbot') return false;
    if (!isLoggedIn()) return false;
    const user = getUser();
    return user && user.role === 'citizen';
  }

  /* === DOM Injection ===================================================== */

  function injectWidget() {
    /* FAB */
    fab = document.createElement('button');
    fab.className = 'chat-widget-fab';
    fab.setAttribute('aria-label', 'Open chat assistant');
    fab.innerHTML =
      '<span class="icon">smart_toy</span>' +
      '<span class="chat-widget-fab-badge" id="cwBadge"></span>';
    fab.addEventListener('click', togglePanel);
    document.body.appendChild(fab);
    badge = fab.querySelector('#cwBadge');

    /* Panel */
    panel = document.createElement('div');
    panel.className = 'chat-widget-panel';
    panel.innerHTML = `
      <div class="chat-widget-header">
        <div class="chat-widget-header-icon"><span class="icon">smart_toy</span></div>
        <div class="chat-widget-header-info">
          <div class="chat-widget-header-title">${BOT_NAME}</div>
          <div class="chat-widget-header-subtitle">PR&DW Assistant</div>
        </div>
        <div class="chat-widget-header-actions">
          <select class="chat-widget-lang" id="cwLang" aria-label="Language">
            <option value="english">EN</option>
            <option value="odia">OD</option>
            <option value="hindi">HI</option>
          </select>
          <button class="chat-widget-header-btn" id="cwTransfer" title="Open in full page" aria-label="Open in full page">
            <span class="icon">open_in_new</span>
          </button>
          <button class="chat-widget-header-btn" id="cwClose" title="Close chat" aria-label="Close chat">
            <span class="icon">close</span>
          </button>
        </div>
      </div>
      <div class="chat-widget-messages" id="cwMessages">
        <div class="bot-group">
          <div class="bot-header"><div class="bot-avatar"><span class="icon filled" style="color:#fff;font-size:14px;">smart_toy</span></div><span class="bot-name">${BOT_NAME}</span></div>
          <div class="chat-bubble bot">Hello! I'm <strong>${BOT_NAME}</strong>, your PR&DW assistant. How can I help you today?</div>
        </div>
      </div>
      <div class="chat-file-preview chat-widget-file-preview hidden" id="cwFilePreview"></div>
      <div class="chat-widget-input-area">
        <input type="file" id="cwFileInput" multiple accept="image/*,audio/*,video/*" style="display:none">
        <button class="chat-attach-btn chat-widget-attach-btn" id="cwAttach" aria-label="Attach file"><span class="icon">attach_file</span></button>
        <input type="text" id="cwInput" placeholder="Type a message..." autocomplete="off">
        <button class="chat-send-btn" id="cwSend" aria-label="Send message"><span class="icon">send</span></button>
      </div>`;
    document.body.appendChild(panel);

    /* Grab references */
    messagesEl = document.getElementById('cwMessages');
    inputEl = document.getElementById('cwInput');
    sendBtn = document.getElementById('cwSend');
    langSelect = document.getElementById('cwLang');
    wFileInput = document.getElementById('cwFileInput');
    wFilePreview = document.getElementById('cwFilePreview');

    /* Events */
    document.getElementById('cwClose').addEventListener('click', closePanel);
    document.getElementById('cwTransfer').addEventListener('click', transferToPage);
    document.getElementById('cwAttach').addEventListener('click', () => wFileInput.click());
    sendBtn.addEventListener('click', sendMessage);
    inputEl.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });
    wFileInput.addEventListener('change', () => {
      wAddPendingFiles(Array.from(wFileInput.files));
      wFileInput.value = '';
    });
  }

  /* === Widget File Attachment ============================================= */

  function wAddPendingFiles(files) {
    for (const f of files) {
      if (wPendingFiles.length >= W_MAX_FILES) { showToast('Maximum ' + W_MAX_FILES + ' files', 'error'); break; }
      if (!W_ALLOWED_TYPES.some(t => f.type.startsWith(t))) { showToast(f.name + ': unsupported type', 'error'); continue; }
      if (f.size > W_MAX_FILE_SIZE) { showToast(f.name + ': exceeds 50 MB', 'error'); continue; }
      if (wPendingFiles.some(e => e.name === f.name && e.size === f.size)) continue;
      wPendingFiles.push(f);
    }
    wRenderFilePreview();
  }

  function wRemovePendingFile(idx) {
    wPendingFiles.splice(idx, 1);
    wRenderFilePreview();
  }

  /* Expose to onclick in rendered HTML */
  window._cwRemoveFile = wRemovePendingFile;

  function wRenderFilePreview() {
    if (!wFilePreview) return;
    if (!wPendingFiles.length) { wFilePreview.innerHTML = ''; wFilePreview.classList.add('hidden'); return; }
    wFilePreview.classList.remove('hidden');
    wFilePreview.innerHTML = wPendingFiles.map((f, i) => {
      const isImg = f.type.startsWith('image/');
      const thumb = isImg ? '<img src="' + URL.createObjectURL(f) + '" alt="">' : '<span class="icon" style="font-size:18px;">description</span>';
      return '<div class="chat-file-chip">' + thumb + '<span class="chat-file-chip-name">' + escapeHtml(f.name) + '</span><button onclick="_cwRemoveFile(' + i + ')" class="chat-file-chip-remove" aria-label="Remove"><span class="icon" style="font-size:14px;">close</span></button></div>';
    }).join('');
  }

  async function wUploadPendingFiles() {
    if (!wPendingFiles.length) return;
    const fd = new FormData();
    wPendingFiles.forEach(f => fd.append('files', f));
    const result = await apiFormData('POST', '/chat/upload', fd);
    const newIds = (result.files || []).map(f => f.id);
    // Replace (not accumulate) so retries don't duplicate attachments
    wUploadedAttachmentIds = newIds;
    wPendingFiles = [];
    wRenderFilePreview();
  }

  /* === Panel Toggle ====================================================== */

  function togglePanel() {
    if (panelOpen) closePanel(); else openPanel();
  }

  function openPanel() {
    panelOpen = true;
    panel.classList.remove('closing');
    panel.classList.add('open');
    fab.classList.add('open');
    fab.querySelector('.icon').textContent = 'close';
    unreadCount = 0;
    updateBadge();
    inputEl.focus();
  }

  function closePanel() {
    panelOpen = false;
    fab.classList.remove('open');
    fab.querySelector('.icon').textContent = 'smart_toy';
    panel.classList.add('closing');
    panel.addEventListener('animationend', function handler() {
      panel.removeEventListener('animationend', handler);
      panel.classList.remove('open', 'closing');
    });
  }

  /* === Chat Functions ==================================================== */

  function wAddBubble(role, html) {
    if (role === 'bot') {
      let group = messagesEl.lastElementChild;
      if (!group || !group.classList.contains('bot-group')) {
        group = document.createElement('div');
        group.className = 'bot-group';
        group.innerHTML = `<div class="bot-header"><div class="bot-avatar"><span class="icon filled" style="color:#fff;font-size:14px;">smart_toy</span></div><span class="bot-name">${BOT_NAME}</span></div>`;
        messagesEl.appendChild(group);
      }
      const div = document.createElement('div');
      div.className = 'chat-bubble bot';
      div.innerHTML = html;
      group.appendChild(div);
      messagesEl.scrollTop = messagesEl.scrollHeight;
      return div;
    } else {
      const div = document.createElement('div');
      div.className = `chat-bubble ${role}`;
      div.innerHTML = html;
      messagesEl.appendChild(div);
      messagesEl.scrollTop = messagesEl.scrollHeight;
      return div;
    }
  }

  function splitReply(text) {
    if (!text) return [text];
    const parts = text.split(/\n(?=###\s)|(?:\n\s*\n)/).map(p => p.trim()).filter(Boolean);
    return parts.length <= 1 ? [text] : parts;
  }

  function wSleep(ms) { return new Promise(r => setTimeout(r, ms)); }

  async function wAddBotBubbles(chunks) {
    for (let i = 0; i < chunks.length; i++) {
      if (i > 0) {
        wShowTyping();
        await wSleep(400 + Math.min(chunks[i].length * 4, 800));
        wHideTyping();
      }
      const bubble = wAddBubble('bot', renderMarkdown(chunks[i]));
      bubble.style.opacity = '0';
      bubble.style.transform = 'translateY(6px)';
      requestAnimationFrame(() => {
        bubble.style.transition = 'opacity 0.2s ease, transform 0.2s ease';
        bubble.style.opacity = '1';
        bubble.style.transform = 'translateY(0)';
      });
    }
  }

  function wShowTyping() {
    let group = messagesEl.lastElementChild;
    if (!group || !group.classList.contains('bot-group')) {
      group = document.createElement('div');
      group.className = 'bot-group';
      group.innerHTML = `<div class="bot-header"><div class="bot-avatar"><span class="icon filled" style="color:#fff;font-size:14px;">smart_toy</span></div><span class="bot-name">${BOT_NAME}</span></div>`;
      messagesEl.appendChild(group);
    }
    const div = document.createElement('div');
    div.className = 'chat-bubble bot';
    div.id = 'cwTyping';
    div.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    group.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function wHideTyping() {
    const el = document.getElementById('cwTyping');
    if (el) el.remove();
  }

  function wAddFiledGrievance(fg) {
    let group = messagesEl.lastElementChild;
    if (!group || !group.classList.contains('bot-group')) {
      group = document.createElement('div');
      group.className = 'bot-group';
      group.innerHTML = `<div class="bot-header"><div class="bot-avatar"><span class="icon filled" style="color:#fff;font-size:14px;">smart_toy</span></div><span class="bot-name">${BOT_NAME}</span></div>`;
      messagesEl.appendChild(group);
    }
    const div = document.createElement('div');
    div.className = 'chat-filed-grievance';
    const detailLink = fg.id
      ? `<a href="/grievance-detail?id=${encodeURIComponent(fg.id)}" class="chat-action-btn"><span class="icon">visibility</span> View Details</a>`
      : '';
    div.innerHTML = `
      <h4><span class="icon filled" style="font-size:16px;color:var(--success);">check_circle</span> Grievance Filed Successfully</h4>
      <p><strong>Tracking:</strong> ${escapeHtml(fg.tracking_number)}</p>
      <p>${statusBadge(fg.status)} ${deptBadge(fg.department)} ${priorityBadge(fg.priority)}</p>
      <div class="chat-action-links">
        ${detailLink}
        <a href="/track" class="chat-action-btn secondary"><span class="icon">search</span> Track Grievance</a>
      </div>`;
    group.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function wAddSchemeSources(sources) {
    const schemes = (sources || []).filter(s => s.type === 'scheme' && s.id && s.score > 0.4);
    if (!schemes.length) return;
    let group = messagesEl.lastElementChild;
    if (!group || !group.classList.contains('bot-group')) {
      group = document.createElement('div');
      group.className = 'bot-group';
      group.innerHTML = `<div class="bot-header"><div class="bot-avatar"><span class="icon filled" style="color:#fff;font-size:14px;">smart_toy</span></div><span class="bot-name">${BOT_NAME}</span></div>`;
      messagesEl.appendChild(group);
    }
    const div = document.createElement('div');
    div.className = 'chat-scheme-sources';
    const links = schemes.map(s =>
      `<a href="/scheme-detail?id=${encodeURIComponent(s.id)}" class="chat-action-btn"><span class="icon">open_in_new</span> ${escapeHtml(s.name)}</a>`
    ).join('');
    div.innerHTML = `<div class="chat-scheme-sources-label"><span class="icon">auto_awesome</span> Related Schemes</div><div class="chat-action-links">${links}</div>`;
    group.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  async function sendMessage() {
    const text = inputEl.value.trim();
    const hasFiles = wPendingFiles.length > 0;
    if (!text && !hasFiles) return;
    inputEl.value = '';

    const fileCount = wPendingFiles.length;
    const attachHtml = fileCount ? '<div class="chat-attach-indicator"><span class="icon" style="font-size:14px;">attach_file</span> ' + fileCount + ' file' + (fileCount > 1 ? 's' : '') + ' attached</div>' : '';
    wAddBubble('user', escapeHtml(text || '(files attached)') + attachHtml);
    const historyContent = fileCount
      ? (text || '(files attached)') + ` [${fileCount} file(s) attached]`
      : text;
    chatHistory.push({ role: 'user', content: historyContent });
    wShowTyping();
    sendBtn.disabled = true;

    try {
      if (hasFiles) {
        await wUploadPendingFiles();
      }
      const data = await api('POST', '/chat', {
        message: text || 'I have attached files for my grievance.',
        language: langSelect.value,
        conversation_history: chatHistory.slice(-10),
        attachment_ids: wUploadedAttachmentIds,
      });
      wHideTyping();
      const chunks = splitReply(data.reply);
      await wAddBotBubbles(chunks);
      chatHistory.push({ role: 'assistant', content: data.reply });
      if (data.sources) wAddSchemeSources(data.sources);
      if (data.filed_grievance) {
        wAddFiledGrievance(data.filed_grievance);
        wUploadedAttachmentIds = [];
      }

      if (!panelOpen) {
        unreadCount++;
        updateBadge();
      }
    } catch (e) {
      wHideTyping();
      wAddBubble('bot', 'Sorry, something went wrong. Please try again.');
    }
    sendBtn.disabled = false;
    inputEl.focus();
    saveState();
  }

  /* === Unread Badge ====================================================== */

  function updateBadge() {
    if (!badge) return;
    if (unreadCount > 0) {
      badge.textContent = unreadCount > 9 ? '9+' : unreadCount;
      badge.classList.add('visible');
    } else {
      badge.classList.remove('visible');
    }
  }

  /* === Persistence (sessionStorage) ====================================== */

  function saveState() {
    try {
      sessionStorage.setItem(WIDGET_STORAGE_KEY, JSON.stringify({
        chatHistory: chatHistory,
        messagesHTML: messagesEl.innerHTML,
        lang: langSelect.value,
      }));
    } catch (e) { /* quota exceeded — ignore */ }
  }

  function restoreState() {
    try {
      const raw = sessionStorage.getItem(WIDGET_STORAGE_KEY);
      if (!raw) return;
      const state = JSON.parse(raw);
      if (state.chatHistory && state.chatHistory.length > 0) {
        chatHistory = state.chatHistory;
      }
      if (state.messagesHTML) {
        messagesEl.innerHTML = state.messagesHTML;
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }
      if (state.lang && langSelect) {
        langSelect.value = state.lang;
      }
    } catch (e) { /* corrupted — ignore */ }
  }

  /* === Transfer to Full Page ============================================= */

  function transferToPage() {
    try {
      sessionStorage.setItem(TRANSFER_KEY, JSON.stringify({
        chatHistory: chatHistory,
        messagesHTML: messagesEl.innerHTML,
        lang: langSelect.value,
      }));
      sessionStorage.removeItem(WIDGET_STORAGE_KEY);
    } catch (e) { /* ignore */ }
    window.location.href = '/chatbot';
  }

})();
