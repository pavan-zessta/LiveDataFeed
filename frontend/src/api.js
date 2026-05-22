import axios from "axios";

const BASE = "http://localhost:8000";

export const getFeed = (userId, k = 20) =>
  axios.get(`${BASE}/feed/${userId}`, { params: { k } }).then((r) => r.data);

export const getUser = (userId) =>
  axios.get(`${BASE}/users/${userId}`).then((r) => r.data);

export const createUser = (username, interests) =>
  axios.post(`${BASE}/users`, { username, interests }).then((r) => r.data);

export const createPost = (payload) =>
  axios.post(`${BASE}/posts`, payload).then((r) => r.data);

export const engage = (userId, postId, action) =>
  axios.post(`${BASE}/engage`, { user_id: userId, post_id: postId, action }).then((r) => r.data);
