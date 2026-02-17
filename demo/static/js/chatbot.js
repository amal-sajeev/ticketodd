/* Chatbot JS */
const chatHistory = [];
const messagesEl = document.getElementById('chatMessages');
const inputEl = document.getElementById('chatInput');
const BOT_NAME = 'Seva';

if (!requireAuth()) {}

/* ── File attachment state ── */
let pendingFiles = [];
let uploadedAttachmentIds = [];
const filePreviewEl = document.getElementById('chatFilePreview');
const fileInputEl = document.getElementById('chatFileInput');
const CHAT_ALLOWED_TYPES = ['image/', 'audio/', 'video/'];
const CHAT_MAX_FILE_SIZE = 50 * 1024 * 1024;
const CHAT_MAX_FILES = 5;

if (fileInputEl) {
  fileInputEl.addEventListener('change', () => {
    addPendingFiles(Array.from(fileInputEl.files));
    fileInputEl.value = '';
  });
}

function addPendingFiles(files) {
  for (const f of files) {
    if (pendingFiles.length >= CHAT_MAX_FILES) { showToast(`Maximum ${CHAT_MAX_FILES} files`, 'error'); break; }
    if (!CHAT_ALLOWED_TYPES.some(t => f.type.startsWith(t))) { showToast(`${f.name}: unsupported type`, 'error'); continue; }
    if (f.size > CHAT_MAX_FILE_SIZE) { showToast(`${f.name}: exceeds 50 MB`, 'error'); continue; }
    if (pendingFiles.some(e => e.name === f.name && e.size === f.size)) continue;
    pendingFiles.push(f);
  }
  renderFilePreview();
}

function removePendingFile(idx) {
  pendingFiles.splice(idx, 1);
  renderFilePreview();
}

function renderFilePreview() {
  if (!filePreviewEl) return;
  if (!pendingFiles.length) { filePreviewEl.innerHTML = ''; filePreviewEl.classList.add('hidden'); return; }
  filePreviewEl.classList.remove('hidden');
  filePreviewEl.innerHTML = pendingFiles.map((f, i) => {
    const isImg = f.type.startsWith('image/');
    const thumb = isImg ? `<img src="${URL.createObjectURL(f)}" alt="">` : `<span class="icon" style="font-size:20px;">description</span>`;
    return `<div class="chat-file-chip">${thumb}<span class="chat-file-chip-name">${escapeHtml(f.name)}</span><button onclick="removePendingFile(${i})" class="chat-file-chip-remove" aria-label="Remove"><span class="icon" style="font-size:14px;">close</span></button></div>`;
  }).join('');
}

async function uploadPendingFiles() {
  if (!pendingFiles.length) return;
  const fd = new FormData();
  pendingFiles.forEach(f => fd.append('files', f));
  const result = await apiFormData('POST', '/chat/upload', fd);
  const newIds = (result.files || []).map(f => f.id);
  uploadedAttachmentIds = newIds;
  pendingFiles = [];
  renderFilePreview();
}

/* Restore conversation transferred from the floating widget */
(function restoreTransfer() {
  try {
    const raw = sessionStorage.getItem('chatbot_transfer');
    if (!raw) return;
    const state = JSON.parse(raw);
    sessionStorage.removeItem('chatbot_transfer');
    sessionStorage.removeItem('chatWidgetState');
    if (state.chatHistory && state.chatHistory.length > 0) {
      chatHistory.push(...state.chatHistory);
    }
    if (state.messagesHTML && messagesEl) {
      messagesEl.innerHTML = state.messagesHTML;
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
    if (state.lang) {
      const langEl = document.getElementById('chatLang');
      if (langEl) langEl.value = state.lang;
    }
  } catch (e) { /* corrupted data — ignore */ }
})();

inputEl?.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } });

/* Check if previous sibling already has the bot header so we don't repeat it */
function needsBotHeader() {
  const children = messagesEl.children;
  if (!children.length) return true;
  const last = children[children.length - 1];
  // If last element is a bot bubble or a bot-group, skip header
  if (last.classList.contains('bot-group')) return false;
  return true;
}

function addBubble(role, html) {
  if (role === 'bot') {
    // Group consecutive bot messages under one header
    let group = messagesEl.lastElementChild;
    if (!group || !group.classList.contains('bot-group')) {
      group = document.createElement('div');
      group.className = 'bot-group';
      group.innerHTML = `<div class="bot-header"><div class="bot-avatar"><span class="icon filled" style="color:#fff;font-size:16px;">smart_toy</span></div><span class="bot-name">${BOT_NAME}</span></div>`;
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

/* Split a reply into logical chunks for multi-bubble rendering.
   Splits on markdown headings (###) or double-newlines, keeping
   each chunk meaningful. Short replies stay as one bubble. */
function splitReply(text) {
  if (!text) return [text];
  // Split on markdown headings or double-newline paragraph breaks
  const parts = text.split(/\n(?=###\s)|(?:\n\s*\n)/).map(p => p.trim()).filter(Boolean);
  if (parts.length <= 1) return [text];
  return parts;
}

/* Render multiple bot bubbles with a staggered delay */
async function addBotBubbles(chunks) {
  for (let i = 0; i < chunks.length; i++) {
    if (i > 0) {
      // Show typing briefly between bubbles
      showTyping();
      await sleep(400 + Math.min(chunks[i].length * 4, 800));
      hideTyping();
    }
    const bubble = addBubble('bot', renderMarkdown(chunks[i]));
    bubble.style.opacity = '0';
    bubble.style.transform = 'translateY(6px)';
    requestAnimationFrame(() => {
      bubble.style.transition = 'opacity 0.2s ease, transform 0.2s ease';
      bubble.style.opacity = '1';
      bubble.style.transform = 'translateY(0)';
    });
  }
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function addFiledGrievance(fg) {
  let group = messagesEl.lastElementChild;
  if (!group || !group.classList.contains('bot-group')) {
    group = document.createElement('div');
    group.className = 'bot-group';
    group.innerHTML = `<div class="bot-header"><div class="bot-avatar"><span class="icon filled" style="color:#fff;font-size:16px;">smart_toy</span></div><span class="bot-name">${BOT_NAME}</span></div>`;
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

function addSchemeSources(sources) {
  const schemes = (sources || []).filter(s => s.type === 'scheme' && s.id && s.score > 0.4);
  if (!schemes.length) return;
  let group = messagesEl.lastElementChild;
  if (!group || !group.classList.contains('bot-group')) {
    group = document.createElement('div');
    group.className = 'bot-group';
    group.innerHTML = `<div class="bot-header"><div class="bot-avatar"><span class="icon filled" style="color:#fff;font-size:16px;">smart_toy</span></div><span class="bot-name">${BOT_NAME}</span></div>`;
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

function showTyping() {
  // Ensure there's a bot-group to place the indicator in
  let group = messagesEl.lastElementChild;
  if (!group || !group.classList.contains('bot-group')) {
    group = document.createElement('div');
    group.className = 'bot-group';
    group.innerHTML = `<div class="bot-header"><div class="bot-avatar"><span class="icon filled" style="color:#fff;font-size:16px;">smart_toy</span></div><span class="bot-name">${BOT_NAME}</span></div>`;
    messagesEl.appendChild(group);
  }
  const div = document.createElement('div');
  div.className = 'chat-bubble bot';
  div.id = 'typingIndicator';
  div.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
  group.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}
function hideTyping() { document.getElementById('typingIndicator')?.remove(); }

async function sendMessage() {
  const text = inputEl.value.trim();
  const hasFiles = pendingFiles.length > 0;
  if (!text && !hasFiles) return;
  inputEl.value = '';

  const fileCount = pendingFiles.length;
  const attachHtml = fileCount ? `<div class="chat-attach-indicator"><span class="icon" style="font-size:14px;">attach_file</span> ${fileCount} file${fileCount > 1 ? 's' : ''} attached</div>` : '';
  addBubble('user', escapeHtml(text || '(files attached)') + attachHtml);
  const historyContent = fileCount
    ? (text || '(files attached)') + ` [${fileCount} file(s) attached]`
    : text;
  chatHistory.push({ role: 'user', content: historyContent });
  showTyping();
  document.getElementById('sendBtn').disabled = true;

  try {
    if (hasFiles) {
      await uploadPendingFiles();
    }
    const data = await api('POST', '/chat', {
      message: text || 'I have attached files for my grievance.',
      language: document.getElementById('chatLang').value,
      conversation_history: chatHistory.slice(-10),
      attachment_ids: uploadedAttachmentIds,
    });
    hideTyping();
    const chunks = splitReply(data.reply);
    await addBotBubbles(chunks);
    chatHistory.push({ role: 'assistant', content: data.reply });
    if (data.sources) addSchemeSources(data.sources);
    if (data.filed_grievance) {
      addFiledGrievance(data.filed_grievance);
      uploadedAttachmentIds = [];
    }
  } catch (e) {
    hideTyping();
    addBubble('bot', 'Sorry, something went wrong. Please try again.');
  }
  document.getElementById('sendBtn').disabled = false;
  inputEl.focus();
}
