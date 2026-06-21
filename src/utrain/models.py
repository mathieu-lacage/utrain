import sqlalchemy

metadata = sqlalchemy.MetaData()

projects = sqlalchemy.Table(
    "projects",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Text, primary_key=True),
    sqlalchemy.Column("name", sqlalchemy.Text, nullable=False),
    sqlalchemy.Column("preset_name", sqlalchemy.Text, nullable=False),
    sqlalchemy.Column("config", sqlalchemy.Text, nullable=False),
    sqlalchemy.Column("created_at", sqlalchemy.Float, nullable=False),
)

runs = sqlalchemy.Table(
    "runs",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Text, primary_key=True),
    sqlalchemy.Column("project_id", sqlalchemy.Text, nullable=False),
    sqlalchemy.Column("run_dir", sqlalchemy.Text, nullable=False),
    sqlalchemy.Column("status", sqlalchemy.Text, nullable=False, default="pending"),
    sqlalchemy.Column("pid", sqlalchemy.Integer, nullable=True),
    sqlalchemy.Column("started_at", sqlalchemy.Float, nullable=True),
    sqlalchemy.Column("ended_at", sqlalchemy.Float, nullable=True),
)

serve_processes = sqlalchemy.Table(
    "serve_processes",
    metadata,
    sqlalchemy.Column("run_id", sqlalchemy.Text, primary_key=True),
    sqlalchemy.Column("port", sqlalchemy.Integer, nullable=False),
    sqlalchemy.Column("pid", sqlalchemy.Integer, nullable=False),
    sqlalchemy.Column("started_at", sqlalchemy.Float, nullable=False),
)
