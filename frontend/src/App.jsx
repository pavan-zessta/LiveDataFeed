import { useState, useCallback } from "react";
import { getFeed, getUser, createPost, engage } from "./api";
import "./App.css";

const TOPIC_COLORS = {
  tech: "#1d9bf0", science: "#7856ff", sports: "#00ba7c",
  politics: "#ff7a00", music: "#f91880", food: "#ffd700",
  gaming: "#a855f7", art: "#ec4899", finance: "#22c55e", travel: "#06b6d4",
};

const AVATAR_COLORS = [
  "#1d9bf0","#7856ff","#00ba7c","#ff7a00","#f91880",
  "#ffd700","#a855f7","#ec4899","#22c55e","#06b6d4",
];

function avatarColor(str) {
  let h = 0;
  for (let i = 0; i < str.length; i++) h = str.charCodeAt(i) + ((h << 5) - h);
  return AVATAR_COLORS[Math.abs(h) % AVATAR_COLORS.length];
}

function PostCard({ post, onEngage }) {
  const [liked, setLiked] = useState(false);
  const [reposted, setReposted] = useState(false);

  const handleLike = () => {
    setLiked((v) => !v);
    onEngage(post.post_id, liked ? "click" : "like");
  };

  const handleRepost = () => {
    setReposted((v) => !v);
    onEngage(post.post_id, reposted ? "click" : "repost");
  };

  const score = post.final_score;
  const scoreClass = score >= 0 ? "positive" : "negative";
  const color = avatarColor(post.author_id);
  const topicColor = TOPIC_COLORS[post.primary_topic] || "#71767b";

  return (
    <div className="post-card">
      <div className="post-header">
        <div className="avatar" style={{ background: color }}>
          {post.author_id.slice(0, 2).toUpperCase()}
        </div>
        <div className="post-meta">
          <span className="author">{post.author_id.slice(0, 12)}…</span>
          <span className="topic-badge" style={{ background: topicColor + "22", color: topicColor }}>
            {post.primary_topic}
          </span>
          <div className="post-source">
            via {post.source === "thunder" ? "⚡ Thunder" : "🔭 Phoenix"}
          </div>
        </div>
      </div>

      <p className="post-text">{post.text}</p>

      {post.has_media && <div className="post-media">🖼 Media attachment</div>}

      <div className="post-actions">
        <button className={`action-btn ${liked ? "liked" : ""}`} onClick={handleLike}>
          {liked ? "♥" : "♡"} {post.like_count + (liked ? 1 : 0)}
        </button>
        <button className={`action-btn ${reposted ? "reposted" : ""}`} onClick={handleRepost}>
          ↺ {post.repost_count + (reposted ? 1 : 0)}
        </button>
        <button className="action-btn" onClick={() => onEngage(post.post_id, "reply")}>
          ○ {post.reply_count}
        </button>
      </div>

      <div className="score-bar">
        Score:
        <span className={`score-pill ${scoreClass}`}>
          {score >= 0 ? "+" : ""}{score.toFixed(2)}
        </span>
        {post.action_predictions?.like !== undefined && (
          <span className="score-pill">
            P(like) {(post.action_predictions.like * 100).toFixed(1)}%
          </span>
        )}
      </div>
    </div>
  );
}

function Compose({ userId, onPosted }) {
  const [text, setText] = useState("");
  const [topic, setTopic] = useState("tech");
  const [loading, setLoading] = useState(false);

  const handlePost = async () => {
    if (!text.trim()) return;
    setLoading(true);
    try {
      await createPost({ author_id: userId, text, primary_topic: topic });
      setText("");
      onPosted();
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="compose">
      <div className="avatar" style={{ background: avatarColor(userId) }}>
        {userId.slice(0, 2).toUpperCase()}
      </div>
      <div style={{ flex: 1 }}>
        <textarea
          rows={2}
          placeholder="What's happening?"
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <div className="compose-footer">
          <select value={topic} onChange={(e) => setTopic(e.target.value)}>
            {Object.keys(TOPIC_COLORS).map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
          <button className="btn btn-primary btn-sm" onClick={handlePost} disabled={loading}>
            {loading ? "Posting…" : "Post"}
          </button>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [selectedUser, setSelectedUser] = useState(null);
  const [feed, setFeed] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [userIdInput, setUserIdInput] = useState("");

  const loadFeed = useCallback(async (userId) => {
    if (!userId) return;
    setLoading(true);
    setError(null);
    try {
      const [feedData, userData] = await Promise.all([
        getFeed(userId, 20),
        getUser(userId),
      ]);
      setFeed(feedData.feed || []);
      setSelectedUser(userData);
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleEngage = useCallback(async (postId, action) => {
    if (!selectedUser) return;
    try { await engage(selectedUser.user_id, postId, action); } catch (_) {}
  }, [selectedUser]);

  const handleRefresh = () => loadFeed(selectedUser?.user_id || userIdInput);

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="logo">MiniForYou</div>

        {selectedUser && (
          <div className="user-card">
            <h3>@{selectedUser.username}</h3>
            <p>Following {selectedUser.following_count} users</p>
            <p style={{ marginTop: 8, fontSize: 12 }}>
              Top interest:{" "}
              {Object.entries(selectedUser.interest_vector || {})
                .sort((a, b) => b[1] - a[1])[0]?.[0] || "—"}
            </p>
          </div>
        )}

        <div style={{ marginBottom: 12 }}>
          <input
            style={{
              width: "100%", background: "#16181c", border: "1px solid #2f3336",
              borderRadius: 8, color: "#e7e9ea", padding: "8px 12px",
              fontSize: 13, marginBottom: 8, outline: "none",
            }}
            placeholder="Paste a user_id…"
            value={userIdInput}
            onChange={(e) => setUserIdInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && loadFeed(userIdInput)}
          />
          <button className="btn btn-primary" onClick={() => loadFeed(userIdInput)} disabled={loading}>
            {loading ? "Loading…" : "Load Feed"}
          </button>
        </div>

        {selectedUser && (
          <button className="btn btn-outline" style={{ width: "100%", marginBottom: 16 }} onClick={handleRefresh}>
            ↻ Refresh Feed
          </button>
        )}

        <div style={{ fontSize: 12, color: "#71767b", lineHeight: 1.6 }}>
          <p>Get user IDs from:</p>
          <code style={{ color: "#1d9bf0", fontSize: 11 }}>data/generated/dataset.json</code>
        </div>
      </aside>

      <main className="main">
        <div className="feed-header">
          {selectedUser ? `For You — @${selectedUser.username}` : "For You"}
        </div>

        {selectedUser && <Compose userId={selectedUser.user_id} onPosted={handleRefresh} />}

        {error && <div className="error-banner">⚠ {error}</div>}
        {loading && <div className="spinner" />}

        {!loading && feed.length === 0 && !error && (
          <div className="empty-state">
            <h3>No feed yet</h3>
            <p>Paste a user_id and click Load Feed</p>
          </div>
        )}

        {!loading && feed.map((post) => (
          <PostCard key={post.post_id} post={post} onEngage={handleEngage} />
        ))}
      </main>
    </div>
  );
}
