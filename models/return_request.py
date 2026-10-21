from datetime import datetime, timezone

from extensions import db


class ReturnRequest(db.Model):
    __tablename__ = "return_requests"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False,
        index=True
    )

    product_id = db.Column(db.String(100), nullable=False, index=True)
    product_name = db.Column(db.String(200), nullable=False)

    selected_reason = db.Column(
        db.String(100),
        nullable=False,
        index=True
    )

    reason_comment = db.Column(db.Text, nullable=False)

    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True
    )

    analysis_status = db.Column(
        db.String(30),
        nullable=False,
        default="Pending",
        index=True
    )

    user = db.relationship(
        "User",
        backref=db.backref("return_requests", lazy="select")
    )

    images = db.relationship(
        "ReturnImage",
        back_populates="return_request",
        cascade="all, delete-orphan",
        lazy="select"
    )

    def __repr__(self):
        return f"&lt;ReturnRequest {self.id}: {self.product_name}&gt;"