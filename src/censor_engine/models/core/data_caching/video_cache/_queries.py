CREATE_FRAMES_TABLE = """
CREATE TABLE IF NOT EXISTS frames (
    frame INTEGER,
    model TEXT,
    data TEXT,
    PRIMARY KEY (frame, model)
);
"""

UPSERT_FRAME = """
INSERT INTO frames(frame, model, data)
VALUES (?, ?, ?)
ON CONFLICT(frame, model)
DO UPDATE SET data=excluded.data
"""

GET_FRAME = """
SELECT data
FROM frames
WHERE frame=? AND model=?
"""

FRAME_EXISTS = """
SELECT 1
FROM frames
WHERE frame=? AND model=?
"""
