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
        """
        Validate rating, generate feedback_id, insert into DB.
        Returns feedback_id as f"fb_{uuid4().hex[:8]}".
        Raises ValueError if rating not in [1, 5].
        """
        if not (1 <= rating <= 5):
            raise ValueError(f"Rating must be between 1 and 5. Got: {rating}")

        # spec: f"fb_{uuid4().hex[:8]}"
        feedback_id = f"fb_{uuid4().hex[:8]}"
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
        """
        Returns FeedbackSummary with avg_rating and total_count for a response.
        """
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

    def get_low_rated_responses(
        self, threshold: float = 2.5, limit: int = 10
    ) -> list[dict]:
        """
        Returns responses with avg feedback rating below threshold.
        Keys: response_id, query_id, avg_rating, feedback_count.
        Ordered by avg_rating ascending.
        """
        sql = """
            SELECT
                f.response_id,
                r.query_id,
                AVG(f.rating)::FLOAT AS avg_rating,
                COUNT(f.id) AS feedback_count
            FROM intellisupport.feedback f
            JOIN intellisupport.responses r ON r.response_id = f.response_id
            GROUP BY f.response_id, r.query_id
            HAVING AVG(f.rating) < %s
            ORDER BY avg_rating ASC
            LIMIT %s
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (threshold, limit))
            rows = cur.fetchall()
        return [
            {
                "response_id": r[0],
                "query_id": r[1],
                "avg_rating": float(r[2]),
                "feedback_count": int(r[3]),
            }
            for r in rows
        ]
