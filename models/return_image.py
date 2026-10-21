
from extensions import db


class ReturnImage(db.Model):
    __tablename__ = "return_images"

    id = db.Column(db.Integer, primary_key=True)

    return_request_id = db.Column(
        db.Integer,
        db.ForeignKey("return_requests.id"),
        nullable=False,
        index=True
    )

    image_type = db.Column(
        db.String(30),
        nullable=False
    )

    storage_name = db.Column(
        db.String(255),
        nullable=False,
        unique=True
    )

    original_filename = db.Column(
        db.String(255),
        nullable=False
    )

    mime_type = db.Column(
        db.String(100),
        nullable=False
    )

    return_request = db.relationship(
        "ReturnRequest",
        back_populates="images"
    )

    def __repr__(self):
        return f"<ReturnImage {self.id}: {self.image_type}>"
