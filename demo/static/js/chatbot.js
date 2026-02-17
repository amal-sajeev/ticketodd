/* Chatbot JS */
const chatHistory = [];
const messagesEl = document.getElementById('chatMessages');
const inputEl = document.getElementById('chatInput');
const BOT_NAME = 'Seva';

if (!requireAuth()) {}

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
  div.innerHTML = `
    <h4><span class="icon filled" style="font-size:16px;color:var(--success);">check_circle</span> Grievance Filed Successfully</h4>
    <p><strong>Tracking:</strong> <a href="/track">${escapeHtml(fg.tracking_number)}</a></p>
    <p>${statusBadge(fg.status)} ${deptBadge(fg.department)} ${priorityBadge(fg.priority)}</p>`;
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
  if (!text) return;
  inputEl.value = '';
  addBubble('user', escapeHtml(text));
  chatHistory.push({ role: 'user', content: text });
  showTyping();
  document.getElementById('sendBtn').disabled = true;

  try {
    const data = await api('POST', '/chat', {
      message: text,
      language: document.getElementById('chatLang').value,
      conversation_history: chatHistory.slice(-10),
    });
    hideTyping();
    const chunks = splitReply(data.reply);
    await addBotBubbles(chunks);
    chatHistory.push({ role: 'assistant', content: data.reply });
    if (data.filed_grievance) addFiledGrievance(data.filed_grievance);
  } catch (e) {
    hideTyping();
    addBubble('bot', 'Sorry, something went wrong. Please try again.');
  }
  document.getElementById('sendBtn').disabled = false;
  inputEl.focus();
}
