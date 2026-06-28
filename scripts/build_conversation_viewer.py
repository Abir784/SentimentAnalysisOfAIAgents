from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_conversations(path: Path) -> tuple[list[dict], int]:
    conversations: list[dict] = []
    malformed_rows = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                malformed_rows += 1
                continue

            messages = item.get("messages") or []
            conversations.append(
                {
                    "id": len(conversations) + 1,
                    "lineNumber": line_number,
                    "timestamp": item.get("timestamp", ""),
                    "modelA": item.get("model_a", "unknown"),
                    "modelB": item.get("model_b", "unknown"),
                    "topic": item.get("topic", "Untitled"),
                    "turnCount": item.get("turn_count", len(messages)),
                    "messages": [
                        {
                            "turn": message.get("turn"),
                            "speaker": message.get("speaker", "unknown"),
                            "message": message.get("message", ""),
                        }
                        for message in messages
                    ],
                }
            )

    return conversations, malformed_rows


def build_metadata(conversations: list[dict], malformed_rows: int) -> dict:
    model_pairs = Counter(
        f"{conversation['modelA']} + {conversation['modelB']}"
        for conversation in conversations
    )
    models = sorted(
        {
            message["speaker"]
            for conversation in conversations
            for message in conversation["messages"]
        }
    )
    topics = Counter(conversation["topic"] for conversation in conversations)

    return {
        "conversationCount": len(conversations),
        "malformedRows": malformed_rows,
        "modelPairs": dict(sorted(model_pairs.items())),
        "models": models,
        "topics": dict(sorted(topics.items())),
    }


def html_escape_json(data: object) -> str:
    return (
        json.dumps(data, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def create_viewer(input_path: Path, output_path: Path) -> None:
    conversations, malformed_rows = load_conversations(input_path)
    if not conversations:
        raise ValueError(f"No conversations found in {input_path}")

    metadata = build_metadata(conversations, malformed_rows)
    conversation_json = html_escape_json(conversations)
    metadata_json = html_escape_json(metadata)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Synthetic Conversation Viewer</title>
  <style>
    :root {{
      --bg: #f3f5f8;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #657282;
      --line: #dbe1e8;
      --accent: #1768ac;
      --accent-soft: #e8f2fb;
      --bubble-a: #eef6ff;
      --bubble-b: #ffffff;
      --shadow: 0 10px 28px rgba(20, 31, 46, 0.08);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background: var(--bg);
      font-family: "Segoe UI", Arial, sans-serif;
    }}
    .app {{
      display: grid;
      grid-template-columns: minmax(320px, 390px) minmax(0, 1fr);
      height: 100vh;
    }}
    aside {{
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
      border-right: 1px solid var(--line);
      background: var(--panel);
      min-width: 0;
    }}
    header {{
      padding: 18px 18px 12px;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 22px;
      letter-spacing: 0;
    }}
    .source {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-top: 14px;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfcfe;
    }}
    .stat strong {{
      display: block;
      font-size: 18px;
    }}
    .stat span {{
      color: var(--muted);
      font-size: 12px;
    }}
    .filters {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      display: grid;
      gap: 10px;
    }}
    .control-row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }}
    input,
    select {{
      width: 100%;
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      color: var(--ink);
      padding: 10px 11px;
      font: inherit;
      outline: none;
    }}
    input:focus,
    select:focus {{
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(23, 104, 172, 0.15);
    }}
    .result-count {{
      color: var(--muted);
      font-size: 13px;
    }}
    .conversation-list {{
      overflow: auto;
      padding: 8px;
    }}
    .conversation-item {{
      width: 100%;
      display: block;
      text-align: left;
      border: 1px solid transparent;
      border-radius: 8px;
      background: transparent;
      padding: 11px;
      cursor: pointer;
      color: var(--ink);
      font: inherit;
    }}
    .conversation-item:hover {{
      background: #f4f8fc;
    }}
    .conversation-item.active {{
      background: var(--accent-soft);
      border-color: #bad8f1;
    }}
    .item-title {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      font-weight: 650;
      margin-bottom: 6px;
    }}
    .item-title span:first-child {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .item-id {{
      color: var(--muted);
      font-size: 12px;
      flex: 0 0 auto;
    }}
    .item-meta,
    .item-preview {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    main {{
      min-width: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      background: var(--bg);
    }}
    .chat-header {{
      padding: 18px 24px;
      background: rgba(255, 255, 255, 0.92);
      border-bottom: 1px solid var(--line);
    }}
    .chat-title {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
    }}
    .chat-title h2 {{
      margin: 0 0 8px;
      font-size: 24px;
      letter-spacing: 0;
      line-height: 1.25;
    }}
    .chat-meta {{
      color: var(--muted);
      line-height: 1.45;
      font-size: 14px;
    }}
    .copy-button {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      color: var(--ink);
      padding: 9px 12px;
      font: inherit;
      cursor: pointer;
      white-space: nowrap;
    }}
    .copy-button:hover {{
      border-color: var(--accent);
      color: var(--accent);
    }}
    .chat-scroll {{
      overflow: auto;
      padding: 28px min(5vw, 54px);
    }}
    .chat-body {{
      max-width: 980px;
      margin: 0 auto;
      display: grid;
      gap: 18px;
    }}
    .message {{
      display: grid;
      grid-template-columns: 44px minmax(0, 1fr);
      gap: 12px;
      align-items: start;
      max-width: 860px;
    }}
    .message:nth-child(even) {{
      justify-self: end;
      grid-template-columns: minmax(0, 1fr) 44px;
    }}
    .message:nth-child(even) .avatar {{
      grid-column: 2;
      grid-row: 1;
    }}
    .message:nth-child(even) .bubble {{
      grid-column: 1;
      grid-row: 1;
      background: var(--bubble-b);
    }}
    .avatar {{
      width: 44px;
      height: 44px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      background: #1f2937;
      color: #ffffff;
      font-weight: 700;
      letter-spacing: 0;
      text-transform: uppercase;
    }}
    .bubble {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--bubble-a);
      padding: 14px 16px;
      box-shadow: var(--shadow);
    }}
    .speaker {{
      display: flex;
      align-items: baseline;
      gap: 8px;
      margin-bottom: 8px;
    }}
    .speaker strong {{
      font-size: 15px;
    }}
    .speaker span {{
      color: var(--muted);
      font-size: 12px;
    }}
    .message-text {{
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      line-height: 1.58;
      font-size: 15px;
    }}
    .empty {{
      color: var(--muted);
      padding: 40px;
      text-align: center;
    }}
    mark {{
      background: #fff2a8;
      color: inherit;
      padding: 0 2px;
      border-radius: 3px;
    }}
    @media (max-width: 820px) {{
      .app {{
        grid-template-columns: 1fr;
        grid-template-rows: minmax(360px, 46vh) minmax(0, 1fr);
      }}
      aside {{
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }}
      .chat-header {{
        padding: 14px 16px;
      }}
      .chat-title {{
        display: grid;
      }}
      .chat-scroll {{
        padding: 18px 12px;
      }}
      .message,
      .message:nth-child(even) {{
        grid-template-columns: 36px minmax(0, 1fr);
        justify-self: stretch;
      }}
      .message:nth-child(even) .avatar {{
        grid-column: 1;
      }}
      .message:nth-child(even) .bubble {{
        grid-column: 2;
      }}
      .avatar {{
        width: 36px;
        height: 36px;
      }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <header>
        <h1>Conversation Viewer</h1>
        <p class="source">Source: {input_path.as_posix()}</p>
        <div class="stats">
          <div class="stat"><strong id="totalConversations">0</strong><span>conversations</span></div>
          <div class="stat"><strong id="totalTopics">0</strong><span>topics</span></div>
          <div class="stat"><strong id="totalModels">0</strong><span>models</span></div>
        </div>
      </header>
      <section class="filters" aria-label="Conversation filters">
        <input id="searchInput" type="search" placeholder="Search topic or message">
        <div class="control-row">
          <select id="topicFilter" aria-label="Filter by topic"></select>
          <select id="modelFilter" aria-label="Filter by model"></select>
        </div>
        <div class="control-row">
          <select id="pairFilter" aria-label="Filter by model pair"></select>
          <select id="sortSelect" aria-label="Sort conversations">
            <option value="id-asc">Oldest first</option>
            <option value="id-desc">Newest first</option>
            <option value="topic-asc">Topic A-Z</option>
          </select>
        </div>
        <div class="result-count" id="resultCount"></div>
      </section>
      <nav class="conversation-list" id="conversationList" aria-label="Conversations"></nav>
    </aside>
    <main>
      <section class="chat-header">
        <div class="chat-title">
          <div>
            <h2 id="chatTopic">Select a conversation</h2>
            <div class="chat-meta" id="chatMeta"></div>
          </div>
          <button class="copy-button" id="copyButton" type="button">Copy conversation</button>
        </div>
      </section>
      <section class="chat-scroll">
        <div class="chat-body" id="chatBody"></div>
      </section>
    </main>
  </div>
  <script>
    const conversations = {conversation_json};
    const metadata = {metadata_json};

    const state = {{
      filtered: [],
      selectedId: conversations[0]?.id ?? null,
      query: "",
    }};

    const elements = {{
      totalConversations: document.getElementById("totalConversations"),
      totalTopics: document.getElementById("totalTopics"),
      totalModels: document.getElementById("totalModels"),
      searchInput: document.getElementById("searchInput"),
      topicFilter: document.getElementById("topicFilter"),
      modelFilter: document.getElementById("modelFilter"),
      pairFilter: document.getElementById("pairFilter"),
      sortSelect: document.getElementById("sortSelect"),
      resultCount: document.getElementById("resultCount"),
      conversationList: document.getElementById("conversationList"),
      chatTopic: document.getElementById("chatTopic"),
      chatMeta: document.getElementById("chatMeta"),
      chatBody: document.getElementById("chatBody"),
      copyButton: document.getElementById("copyButton"),
    }};

    function normalize(value) {{
      return String(value ?? "").toLowerCase();
    }}

    function pairName(conversation) {{
      return `${{conversation.modelA}} + ${{conversation.modelB}}`;
    }}

    function formatDate(value) {{
      if (!value) return "No timestamp";
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return value;
      return date.toLocaleString();
    }}

    function initials(name) {{
      return String(name || "?")
        .split(/[-_\\s]+/)
        .filter(Boolean)
        .slice(0, 2)
        .map((part) => part[0])
        .join("");
    }}

    function escapeHtml(value) {{
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }}

    function highlight(value) {{
      const text = escapeHtml(value);
      const query = state.query.trim();
      if (!query) return text;
      const escapedQuery = query.replace(/[.*+?^${{}}()|[\\]\\\\]/g, "\\\\$&");
      return text.replace(new RegExp(`(${{escapedQuery}})`, "gi"), "<mark>$1</mark>");
    }}

    function populateFilters() {{
      elements.totalConversations.textContent = metadata.conversationCount.toLocaleString();
      elements.totalTopics.textContent = Object.keys(metadata.topics).length.toLocaleString();
      elements.totalModels.textContent = metadata.models.length.toLocaleString();

      const topicOptions = ["<option value=\\"\\">All topics</option>"]
        .concat(Object.entries(metadata.topics).map(([topic, count]) => (
          `<option value="${{escapeHtml(topic)}}">${{escapeHtml(topic)}} (${{count}})</option>`
        )));
      elements.topicFilter.innerHTML = topicOptions.join("");

      const modelOptions = ["<option value=\\"\\">All models</option>"]
        .concat(metadata.models.map((model) => (
          `<option value="${{escapeHtml(model)}}">${{escapeHtml(model)}}</option>`
        )));
      elements.modelFilter.innerHTML = modelOptions.join("");

      const pairOptions = ["<option value=\\"\\">All pairs</option>"]
        .concat(Object.entries(metadata.modelPairs).map(([pair, count]) => (
          `<option value="${{escapeHtml(pair)}}">${{escapeHtml(pair)}} (${{count}})</option>`
        )));
      elements.pairFilter.innerHTML = pairOptions.join("");
    }}

    function matchesConversation(conversation) {{
      const topic = elements.topicFilter.value;
      const model = elements.modelFilter.value;
      const pair = elements.pairFilter.value;
      const query = normalize(elements.searchInput.value.trim());

      if (topic && conversation.topic !== topic) return false;
      if (model && !conversation.messages.some((message) => message.speaker === model)) return false;
      if (pair && pairName(conversation) !== pair) return false;
      if (!query) return true;

      const haystack = normalize([
        conversation.topic,
        conversation.modelA,
        conversation.modelB,
        ...conversation.messages.map((message) => `${{message.speaker}} ${{message.message}}`),
      ].join(" "));
      return haystack.includes(query);
    }}

    function sortConversations(items) {{
      const sortMode = elements.sortSelect.value;
      return items.slice().sort((a, b) => {{
        if (sortMode === "id-desc") return b.id - a.id;
        if (sortMode === "topic-asc") {{
          return a.topic.localeCompare(b.topic) || a.id - b.id;
        }}
        return a.id - b.id;
      }});
    }}

    function renderList() {{
      state.query = elements.searchInput.value.trim();
      state.filtered = sortConversations(conversations.filter(matchesConversation));
      elements.resultCount.textContent = `${{state.filtered.length.toLocaleString()}} of ${{conversations.length.toLocaleString()}} conversations`;

      if (!state.filtered.some((conversation) => conversation.id === state.selectedId)) {{
        state.selectedId = state.filtered[0]?.id ?? null;
      }}

      if (state.filtered.length === 0) {{
        elements.conversationList.innerHTML = '<div class="empty">No conversations match the current filters.</div>';
        renderChat(null);
        return;
      }}

      elements.conversationList.innerHTML = state.filtered.map((conversation) => {{
        const firstMessage = conversation.messages[0]?.message ?? "";
        const active = conversation.id === state.selectedId ? " active" : "";
        return `
          <button class="conversation-item${{active}}" type="button" data-id="${{conversation.id}}">
            <div class="item-title">
              <span>${{highlight(conversation.topic)}}</span>
              <span class="item-id">#${{conversation.id}}</span>
            </div>
            <div class="item-meta">${{escapeHtml(pairName(conversation))}} · ${{formatDate(conversation.timestamp)}}</div>
            <div class="item-preview">${{highlight(firstMessage)}}</div>
          </button>
        `;
      }}).join("");

      renderChat(conversations.find((conversation) => conversation.id === state.selectedId));
    }}

    function renderChat(conversation) {{
      if (!conversation) {{
        elements.chatTopic.textContent = "No conversation selected";
        elements.chatMeta.textContent = "";
        elements.chatBody.innerHTML = '<div class="empty">Adjust the filters to show conversations.</div>';
        return;
      }}

      elements.chatTopic.textContent = conversation.topic;
      elements.chatMeta.textContent = `${{pairName(conversation)}} · Conversation #${{conversation.id}} · ${{formatDate(conversation.timestamp)}} · ${{conversation.turnCount}} turns`;
      elements.chatBody.innerHTML = conversation.messages.map((message, index) => `
        <article class="message">
          <div class="avatar" title="${{escapeHtml(message.speaker)}}">${{escapeHtml(initials(message.speaker))}}</div>
          <div class="bubble">
            <div class="speaker">
              <strong>${{escapeHtml(message.speaker)}}</strong>
              <span>Turn ${{escapeHtml(message.turn ?? index + 1)}}</span>
            </div>
            <p class="message-text">${{highlight(message.message)}}</p>
          </div>
        </article>
      `).join("");
    }}

    function copySelectedConversation() {{
      const conversation = conversations.find((item) => item.id === state.selectedId);
      if (!conversation) return;
      const text = [
        `Topic: ${{conversation.topic}}`,
        `Models: ${{pairName(conversation)}}`,
        `Timestamp: ${{conversation.timestamp}}`,
        "",
        ...conversation.messages.map((message) => `${{message.speaker}}: ${{message.message}}`),
      ].join("\\n");
      navigator.clipboard?.writeText(text).then(() => {{
        elements.copyButton.textContent = "Copied";
        window.setTimeout(() => {{
          elements.copyButton.textContent = "Copy conversation";
        }}, 1200);
      }});
    }}

    elements.conversationList.addEventListener("click", (event) => {{
      const button = event.target.closest(".conversation-item");
      if (!button) return;
      state.selectedId = Number(button.dataset.id);
      renderList();
    }});

    [
      elements.searchInput,
      elements.topicFilter,
      elements.modelFilter,
      elements.pairFilter,
      elements.sortSelect,
    ].forEach((element) => element.addEventListener("input", renderList));

    elements.copyButton.addEventListener("click", copySelectedConversation);

    populateFilters();
    renderList();
  </script>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a local chat-style viewer for synthetic conversations."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/sythetic/conversations.jsonl"),
        help="Path to conversations JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sythetic/conversation_viewer.html"),
        help="Output HTML path.",
    )
    args = parser.parse_args()

    create_viewer(args.input, args.output)
    print(f"Conversation viewer: {args.output}")


if __name__ == "__main__":
    main()
