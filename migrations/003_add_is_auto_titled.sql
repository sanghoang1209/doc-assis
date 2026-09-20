ALTER TABLE sessions 
ADD COLUMN is_auto_titled BOOLEAN DEFAULT FALSE NOT NULL;

UPDATE sessions
SET is_auto_titled = TRUE
WHERE title NOT IN ('New Conversation', 'New Chat', 'Cuộc trò chuyện mới');
