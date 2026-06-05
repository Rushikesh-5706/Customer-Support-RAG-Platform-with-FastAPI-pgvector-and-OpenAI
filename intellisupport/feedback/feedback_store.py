import logging
from uuid import uuid4
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class FeedbackSummary(BaseModel):
    response_id: str
    avg_rating: float
    total_count: int


class FeedbackStore:

    def __init__(self, conn):
        self._conn = conn

    def store_feedback(
        self, response_id: str, rating: int, comment: str = None
    ) -> str:
        if not (1 <= rating <= 5):
            raise ValueError(f"Rating must be between 1 and 5. Got: {rating}")

        feedback_id = f"fb_{uuid4().hex[:16]}"
        sql = """
            INSERT INTO intellisupport.feedback
                (feedback_id, response_id, rating, comment)
            VALUES (%s, %s, %s, %s)
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (feedback_id, response_id, rating, comment))
        self._conn.commit()
        logger.info("Stored feedback '%s' for response '%s'.", feedback_id, response_id)
        return feedback_id

    def get_feedback_summary(self, response_id: str) -> FeedbackSummary:
        sql = """
            SELECT AVG(rating)::FLOAT, COUNT(*)
            FROM intellisupport.feedback
            WHERE response_id = %s
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (response_id,))
            row = cur.fetchone()

        avg_rating = float(row[0]) if row[0] is not None else 0.0
        total_count = int(row[1])
        return FeedbackSummary(
            response_id=response_id,
            avg_rating=avg_rating,
            total_count=total_count,
        )

    def list_low_rated_responses(self, threshold: float = 2.5) -> list[dict]:
        sql = """
            SELECT f.response_id, AVG(f.rating)::FLOAT AS avg_rating, COUNT(*) AS count
            FROM intellisupport.feedback f
            GROUP BY f.response_id
            HAVING AVG(f.rating) <= %s
            ORDER BY avg_rating ASC
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (threshold,))
            rows = cur.fetchall()
        return [
            {"response_id": r[0], "avg_rating": float(r[1]), "count": int(r[2])}
            for r in rows
        ]
